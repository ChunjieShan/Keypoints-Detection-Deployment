from trt_utils import *

context = init_trt_engine("../onehand10k.engine")
for _ in range(50):
    trt_engine_server_inference(context, "../6158.png", False)
    # trt_engine_inference("../onehand10k.engine", "../6158.png")
