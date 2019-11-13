import os

import cv2
import flask
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import jsonify
from flask import request

cwd = os.getcwd()

os.chdir("./ShowAttendAndTellModel/")
print(os.getcwd())
from ShowAttendAndTellModel.run_inference import CaptionInference

os.chdir(cwd)

os.chdir("./ssd_mobilenet_v2_oid_v4_2018_12_12/")
cvNet = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'graph.pbtxt')
classList = [line.rstrip('\n') for line in open("oid_v4_label_map.txt")]
os.chdir(cwd)

app = flask.Flask(__name__)
app.config["DEBUG"] = False
video_capture_dict = {}


@app.route('/objectdetection', methods=['GET'])
def fuck():
    return "fuck"


@app.route('/objectdetection', methods=['POST'])
def predict():
    request_json = request.json
    img = cv2.imdecode(np.frombuffer(request.data, np.uint8), cv2.IMREAD_UNCHANGED)
    height, width, _ = img.shape
    cvNet.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
    cvOut = cvNet.forward()
    output_result_list = []
    for detection in cvOut[0, 0, :, :]:
        score = float(detection[2])
        if score > 0.3:
            left = detection[3] * height
            top = detection[4] * width
            right = detection[5] * height
            bottom = detection[6] * width
            cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
            print(classList[int(detection[1]) - 1])
            cv2.putText(img, classList[int(detection[1]) - 1], (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 1, cv2.LINE_AA)
            output_result_list.append({"className": classList[int(detection[1]) - 1],
                                       "left": (int(left)),
                                       "top": (int(top)),
                                       "right": int(right),
                                       "bottom": int(bottom)
                                       })
    # data["result"] = output_result_list
    return jsonify(output_result_list)


@app.route('/imagecaption', methods=['POST'])
def imagecaption():
    request_json = request.json
    img = cv2.imdecode(np.frombuffer(request.data, np.uint8), cv2.IMREAD_UNCHANGED)
    height, width, _ = img.shape
    img_np = np.array(cap_infer.resize_image(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
                                             cap_infer.model.cnn.image_size)).astype(np.float32)
    if cap_infer.use_inception:
        img_np /= 255.0
        img_np -= 0.5
        img_np *= 2.0
    if img_np.ndim == 2:
        img_np = np.stack((img_np,) * 3, axis=-1)
    alphas, betas, captions = cap_infer.inference_np(np.array([img_np]))
    output_result_list = []
    for alpha, beta, caption in zip(alphas, betas, captions):
        # out_file.write("{}\t{}\n".format(fname, caption))
        # if args.visualize:
        #     visualize(alpha, beta, caption, fname, args.use_inception)
        output_result_list.append({"caption": caption})
    # data["result"] = output_result_list
    return jsonify(output_result_list)


if __name__ == "__main__":
    sess = tf.Session()
    cap_infer = CaptionInference(sess, "ShowAttendAndTellModel/model_best/model-best", use_inception=True)
    app.run(host='0.0.0.0')
