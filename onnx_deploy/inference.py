import matplotlib.pyplot as plt
from onnx_utils import init_onnx_engine, onnx_server_inference

classes = {0: 'wrist',
           1: 'thumb1',
           2: 'thumb2',
           3: 'thumb3',
           4: 'thumb4',
           5: 'forefinger1',
           6: 'forefinger2',
           7: 'forefinger3',
           8: 'forefinger4',
           9: 'middle_finger1',
           10: 'middle_finger2',
           11: 'middle_finger3',
           12: 'middle_finger4',
           13: 'ring_finger1',
           14: 'ring_finger2',
           15: 'ring_finger3',
           16: 'ring_finger4',
           17: 'pinky_finger1',
           18: 'pinky_finger2',
           19: 'pinky_finger3',
           20: 'pinky_finger4'}

session = init_onnx_engine("../ddh.onnx", device="gpu")

for _ in range(50):
    onnx_server_inference(session, img_path="../00001.png", save_dir="./images")
    # img_save_path = do_inference("../onehand10k.onnx", "gpu", "../6158.png", False, "./images")
# print(img_save_path)

lost = []

# for result in results:
#     points, probs = result[0].tolist(), result[1].tolist()[0]
#     vis_pose(img, points[0])
#     plt.imshow(img)
#     plt.show()

# print(probs)

# for i in range(21):
#     if probs[i][0] < 0.3:
#         print(classes[i])
