import sys
import os
import numpy as np
import tensorrt as trt
import time
import matplotlib.pyplot as plt
import pycuda.autoinit
import pycuda.driver as cuda
import cv2 as cv

from collections import OrderedDict
from typing import List

BATCH_SIZE = 1  # batch size
CHANNEL = 3  # image's channel
INPUT_IMG_H = 256  # height
INPUT_IMG_W = 256  # width

KEYPOINT_NUM = 21  # num of keypoint
HEATMAP_HEIGHT = 64  # height of heatmap
HEATMAP_WIDTH = 64  # width of heatmap

INPUT_SHAPE = BATCH_SIZE * CHANNEL * INPUT_IMG_W * INPUT_IMG_H
OUTPUT_SHAPE = BATCH_SIZE * KEYPOINT_NUM * HEATMAP_WIDTH * HEATMAP_HEIGHT


def init_trt_engine(trt_path):
    """
    Initialize tensorrt engine path
    :param trt_path: TensorRT Engine path;
    :return:
    """
    logger = trt.Logger(trt.Logger.INFO)
    with open(trt_path, 'rb') as f, trt.Runtime(logger) as runtime:
        model = runtime.deserialize_cuda_engine(f.read())

    context = model.create_execution_context()

    return context


def do_inference(context,
                 preprocessed_img):
    """
    TensorRT engine inference;
    :param context: TensorRT context;
    :param preprocessed_img: image been preprocessed
    :return:
    """
    host_in = cuda.pagelocked_empty(INPUT_SHAPE, dtype=np.float32)
    np.copyto(host_in, preprocessed_img.ravel())
    host_out = cuda.pagelocked_empty(OUTPUT_SHAPE, dtype=np.float32)

    engine = context.engine
    device_in = cuda.mem_alloc(host_in.nbytes)
    device_out = cuda.mem_alloc(host_out.nbytes)
    bindings = [int(device_in), int(device_out)]
    stream = cuda.Stream()

    start = time.time()
    # for _ in range(100):
    cuda.memcpy_htod_async(device_in, host_in, stream)
    context.execute_async(1, bindings, stream.handle)
    cuda.memcpy_dtoh_async(host_out, device_out, stream)
    stream.synchronize()
    print("{} ms".format((time.time() - start) * 1000))
    return host_out


def trt_engine_inference(trt_context,
                         img_path,
                         show_img: bool = True):
    """
    Initialize an ONNX model and do inference on a preprocessed image;
    :param trt_context: onnx_deploy model path;
    :param img_path: image path;
    :param show_img: show image or not;
    :return:
    """
    img, preprocessed_img = preprocess(img_path, (256, 256))
    if isinstance(trt_context, str):
        _context = init_trt_engine(trt_context)
    elif isinstance(trt_context, trt.IExecutionContext):
        _context = trt_context
    else:
        _context = None

    assert _context is not None, "Your TensorRT IExecution Context is empty!"
    host_out = do_inference(_context, preprocessed_img)

    h, w, _ = img.shape
    center = np.array([[w / 2, h / 2]])
    scale = np.array([[w, h]])
    preds, probs = keypoints_from_heatmaps(host_out, center, scale)

    for pred, prob in zip(preds, probs):
        points, probs = pred.tolist(), prob.tolist()[0]
        vis_pose(img, points)
        if show_img:
            plt.imshow(img)
            plt.show()


def trt_engine_server_inference(context: trt.IExecutionContext,
                                img_path: str,
                                show_img: bool = True):
    """
    TensorRT inference engine inference when launched by Flask.
    It s different from `trt_engine_inference` API. This API won't
    initiate the TensorRT Context, and you need to pass initialized
    TensorRT context to this API;
    :param context: tensorrt.IExecutionContext;
    :param img_path: str;
    :param show_img: bool, show image or not;
    :return:
    """
    trt_engine_inference(context, img_path, show_img)


def preprocess(image_path,
               img_size: tuple = (256, 256)):
    """
    Preprocessing image;
    :param image_path: str;
    :param img_size: tuple, resize size;
    :return:
    """
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_preprocessed = cv.resize(img, img_size)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

    # H * W * C -> C * H * W
    img_preprocessed = np.transpose(img_preprocessed, (2, 0, 1)) / 255.0
    img_preprocessed = np.expand_dims(img_preprocessed, 0)
    img_preprocessed = (img_preprocessed - mean) / std
    # C * H * W -> 1 * C * H * W
    # convert float64 -> float32
    img_preprocessed = img_preprocessed.astype(np.float32)

    return img, img_preprocessed


def get_max_preds(heatmaps: np.array):
    """
    Get keypoint coordinates and probabilities
    :param heatmaps: the heatmap that model predicted
    :return:
    """
    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    prob = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(prob, (1, 1, 2)) > 0.0, preds, -1)
    return preds, prob


def transform_preds(coords: List,
                    center: np.array,
                    scale: np.array,
                    output_size: np.array):
    """
    Get final prediction of keypoint coordinates, and map them back to the image;
    :param coords: ndarray, [N, 2]
    :param center: ndarray, [2]
    :param scale:  ndarray, [2]
    :param output_size: [64, 64]
    :return:
    """
    scale_x = scale[0] / output_size[0]
    scale_y = scale[1] / output_size[1]

    target_coords = np.ones_like(coords)
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5

    return target_coords


def keypoints_from_heatmaps(host_out: List,
                            center: np.array,
                            scale: np.array):
    """
    Post process of the inference
    :param host_out: the output of the tensorrt engine.
    :param center: center of the bounding box;
    :param scale: scale of width and height;
    :return:
    """
    heatmaps = np.array(host_out).reshape((1, 21, 64, 64))
    heatmaps = heatmaps.copy()

    N, K, H, W = heatmaps.shape
    preds, prob = get_max_preds(heatmaps)

    for n in range(N):
        for k in range(K):
            heatmap = heatmaps[n][k]
            px = int(preds[n][k][0])
            py = int(preds[n][k][1])
            if 1 < px < W - 1 and 1 < py < H - 1:
                diff = np.array([
                    heatmap[py][px + 1] - heatmap[py][px - 1],
                    heatmap[py + 1][px] - heatmap[py - 1][px]
                ])
                preds[n][k] += np.sign(diff) * 0.25

    for i in range(N):
        preds[i] = transform_preds(preds[i], center[i], scale[i], [W, H])

    return preds, prob


def vis_pose(img, points):
    for i, point in enumerate(points):
        x, y = point
        x = int(x)
        y = int(y)
        cv.circle(img, (x, y), 4, (0, 0, 255), thickness=-1, lineType=cv.FILLED)
        cv.putText(img, '{}'.format(i), (x, y), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                   color=(255, 255, 255),
                   thickness=1, lineType=cv.LINE_AA)
    return img
