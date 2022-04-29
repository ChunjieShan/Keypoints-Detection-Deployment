import time

import onnxruntime as ort
import cv2 as cv
import numpy as np
from utils import *

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


results = do_inference("./onehand10k.onnx", "./6158.png")
print(results)

lost = []

for result in results:
    points, probs = result[0].tolist(), result[1].tolist()[0]

# print(probs)

for i in range(21):
    if probs[i][0] < 0.3:
        print(classes[i])
