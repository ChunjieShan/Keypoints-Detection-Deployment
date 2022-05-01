import flask
import logging
import os
from flask_cors import CORS
from flask import Flask
from datetime import timedelta
from trt_utils import trt_engine_server_inference, init_trt_engine

app = Flask(__name__)
CORS(app, resources=r'/*')
app.secret_key = 'secret!'

werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.ERROR)

# 解决缓存刷新问题
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)


# 添加header解决跨域
# @app.after_request
# def after_request(response):
#     response.headers['Access-Control-Allow-Origin'] = '*'
#     response.headers['Access-Control-Allow-Credentials'] = 'true'
#     response.headers['Access-Control-Allow-Methods'] = 'POST'
#     response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
#     return response


@app.route('/')
def hello_world():
    return "Hello!"


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    file_path = ""
    if flask.request.method == "POST":
        file_path = flask.request.values.get('fileUrl')
        print(file_path)

    if len(file_path):
        # src_path = os.path.join(app.config['UPLOAD_FOLDER'], file_path)
        # print(src_path)
        # file.save(src_path)
        # shutil.copy(src_path, './temp/uploads')
        # image_path = os.path.join('./temp/uploads', file.filename)
        img_save_path = trt_engine_server_inference(trt_context, file_path, False, "./images/")
        return flask.jsonify({
            'status': 1,
            'img_path': img_save_path
        })

    return flask.jsonify({'status': 0})


if __name__ == '__main__':
    files = [
        'uploads', 'temp/uploads'
    ]
    trt_context = init_trt_engine("../onehand10k.engine")
    for ff in files:
        if not os.path.exists(ff):
            os.makedirs(ff)
    app.run(host='0.0.0.0', port=8080, debug=True)
