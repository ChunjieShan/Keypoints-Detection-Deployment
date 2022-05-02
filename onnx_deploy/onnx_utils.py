import os
import numpy as np
import cv2 as cv
import onnxruntime as ort
import time
import datetime
import matplotlib.pyplot as plt


def init_onnx_engine(onnx_model_path, device=None):
    """
    Initialize ONNX session
    :param device: device you want to run at;
    :param onnx_model_path: str
    :return:
    """
    providers = None
    if device == "gpu":
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            })
        ]
    elif device == "cpu" or None:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(onnx_model_path, providers=providers)
    return sess


def onnx_inference(session: ort.InferenceSession,
                   img: np.array):
    """
    ONNX engine inference;
    :param session: onnxruntime inference session;
    :param img: preprocessed images;
    :return:
    """
    input_tensor = session.get_inputs()
    output_tensor = [node.name for node in session.get_outputs()]
    results = session.run(output_tensor, {input_tensor[0].name: img})

    return results


def do_inference(onnx_session,
                 device: str = None,
                 img_path: str = None,
                 show_img: bool = False,
                 save_dir: str = None):
    """
    Initialize an ONNX model and do inference on a preprocessed images;
    :param onnx_session: onnx_deploy model path, or onnx session
    :param device: device you want to run at, `cpu` or `gpu`.
    :param img_path:
    :param show_img: vis images or not;
    :param save_dir: images save directory;
    :return:
    """
    img, preprocessed_img = preprocess(img_path)

    if isinstance(onnx_session, str):
        sess = init_onnx_engine(onnx_session, device)
    elif isinstance(onnx_session, ort.InferenceSession):
        sess = onnx_session
    else:
        sess = None

    assert sess is not None, "Your ONNXRuntime Session is None!"
    curr_time = time.time()
    results = onnx_inference(sess, preprocessed_img)
    # print(results[0].shape)
    print("{} ms".format((time.time() - curr_time) * 1000))

    h, w, _ = img.shape
    center = np.array([[w / 2, h / 2]])
    scale = np.array([[w, h]])

    results_list = []
    img_save_path = None

    points_list = []
    probs_list = []

    for result in results:
        single_frame_result = keypoints_from_heatmaps(result, center, scale)
        points, probs = single_frame_result[0].tolist(), single_frame_result[1].tolist()
        for point in points:
            vis_pose(img, point)
        points_list.append(points)
        probs_list.append(probs)

        if save_dir is not None:
            assert os.path.exists(save_dir), "Your images save directory do not exist."
            curr_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            img_save_path = os.path.join(save_dir, curr_date + '.jpg')
            # plt.imsave(img_save_path, img)
            cv.imwrite(img_save_path, img)

    return points_list, probs_list


def onnx_server_inference(onnx_session: ort.InferenceSession,
                          img_path: str = None,
                          save_dir: str = None):
    """
    For continuous inference, you need to pass an ONNX Inference Session to this API;
    :param onnx_session:
    :param img_path:
    :param save_dir:
    :return:
    """
    save_path = do_inference(onnx_session, img_path=img_path, show_img=False, save_dir=save_dir)

    return save_path


def preprocess(image_path,
               img_size: tuple = (256, 256)):
    """
    Preprocessing images;
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
    preds[:, :, 0] = preds[:, :, 0] % W  # get x coordinate(width)
    preds[:, :, 1] = preds[:, :, 1] // W  # get y coordinate(height)

    preds = np.where(np.tile(prob, (1, 1, 2)) > 0.0, preds, -1)
    return preds, prob


def transform_preds(coords: np.array,
                    center: np.array,
                    scale: np.array,
                    output_size: np.array):
    """
    Get final prediction of keypoint coordinates, and map them back to the images;
    :param coords: ndarray, [N, 2]
    :param center: bounding box of RoI, ndarray, [2];
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


def keypoints_from_heatmaps(heatmaps: np.array,
                            center: np.array,
                            scale: np.array):
    """
    Post process of the inference
    :param heatmaps: heatmaps that model predicted;
    :param center: center of the bounding box;
    :param scale: scale of width and height;
    :return:
    """
    heatmaps = heatmaps.copy()

    # N: batch size
    # K: num of keypoint
    # H: height
    # W: weight
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
                ])  # differentiation
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
