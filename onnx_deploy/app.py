import flask
import logging
import os
from flask_cors import CORS
from flask import Flask
from datetime import timedelta
from onnx_utils import init_onnx_engine, onnx_server_inference

app = Flask(__name__)
CORS(app, resources=r'/*')
app.secret_key = 'secret!'

werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.ERROR)

# 解决缓存刷新问题
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)


# 添加header解决跨域
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
    return response


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
        points_list, probs_list = onnx_server_inference(onnx_session, file_path, "./images/")
        return flask.jsonify({
            'status': 1,
            'points': points_list,
            'probs': probs_list
        })

    return flask.jsonify({'status': 0})


if __name__ == '__main__':
    files = [
        'uploads', 'temp/uploads'
    ]
    onnx_session = init_onnx_engine("../onehand10k.onnx")
    for ff in files:
        if not os.path.exists(ff):
            os.makedirs(ff)
    app.run(host='0.0.0.0', port=5000, debug=True)
