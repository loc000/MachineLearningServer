import os
import socket

import cv2
import flask
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import jsonify
from flask import request
from zeroconf import ServiceInfo, Zeroconf

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

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("192.168.43.1", 80))
except:
    print("Fuck")
    s.connect(("192.168.0.1", 80))
my_ip = s.getsockname()[0]
print(my_ip)


@app.route('/objectdetection', methods=['GET'])
def fuck():
    return "fuck"


@app.route('/objectdetection', methods=['POST'])
def predict():
    request_json = request.json
    img = cv2.imdecode(np.frombuffer(request.data, np.uint8), cv2.IMREAD_UNCHANGED)
    height, width, _ = img.shape
    rows = img.shape[0]
    cols = img.shape[1]
    cvNet.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
    cvOut = cvNet.forward()
    output_result_list = []
    for detection in cvOut[0, 0, :, :]:
        score = float(detection[2])
        if score > 0.3:
            left = max(detection[3] * cols, 0)
            top = max(detection[4] * rows, 0)
            right = detection[5] * cols
            bottom = detection[6] * rows
            # cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
            # print(classList[int(detection[1]) - 1])
            # cv2.putText(img, classList[int(detection[1]) - 1], (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 1,
            #             (0, 255, 255), 1, cv2.LINE_AA)

            mid_x, mid_y = left + (right - left) / 2, top + (bottom - top) / 2
            loc = ""
            if mid_y < height / 3:
                loc += "Top "
            elif mid_y > height / 3 * 2:
                loc += "Bottom "
            if mid_x < width / 3:
                loc += "left"
            elif mid_x > width / 3 * 2:
                loc += "right"
            if loc == "":
                loc = "Center"
            # cv2.putText(img, loc, (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 1,
            #             (0, 255, 255), 1, cv2.LINE_AA)

            output_result_list.append({"className": classList[int(detection[1]) - 1],
                                       "left": (int(left)),
                                       "top": (int(top)),
                                       "right": int(right),
                                       "bottom": int(bottom),
                                       "location_description": loc
                                       })
    # data["result"] = output_result_list
    # cv2.imshow("loc",img)
    # cv2.waitKey(1)
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
    object_detection_info = ServiceInfo("_oml._tcp.local.",
                                        "_oml._tcp.local.",
                                        socket.inet_aton(my_ip), 5000, 0, 0,
                                        {'type': 'objectdetection'},
                                        "object_detection_machine_learning_server.tcp.local.")

    image_caption_info = ServiceInfo("_icml._tcp.local.",
                                     "_icml._tcp.local.",
                                     socket.inet_aton(my_ip), 5000, 0, 0,
                                     {'type': 'imagecaption'},
                                     "image_caption_machine_learning_server.tcp.local.")

    zeroconf = Zeroconf()
    zeroconf.register_service(object_detection_info)
    zeroconf.register_service(image_caption_info)
    app.run(host='0.0.0.0')
