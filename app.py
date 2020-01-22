import os
import time
import cv2
import flask
import numpy as np
# from PIL import Image
from flask import jsonify
from flask import request
import wget


cwd = os.getcwd()
print(cwd)
print(os.listdir("ssd_mobilenet_v2_oid_v4_2018_12_12"))

# os.chdir("./ssd_mobilenet_v2_oid_v4_2018_12_12/")
wget.download("https://github.com/loc000/MachineLearningServer/releases/download/ssd_mobilenet_v2_oid_v4_2018_12_12/frozen_inference_graph.pb",'ssd_mobilenet_v2_oid_v4_2018_12_12/frozen_inference_graph.pb')
cvNet = cv2.dnn.readNetFromTensorflow('ssd_mobilenet_v2_oid_v4_2018_12_12/frozen_inference_graph.pb', 'ssd_mobilenet_v2_oid_v4_2018_12_12/graph.pbtxt')
classList = [line.rstrip('\n') for line in open("ssd_mobilenet_v2_oid_v4_2018_12_12/oid_v4_label_map.txt")]
# os.chdir(cwd)

app = flask.Flask(__name__)
app.config["DEBUG"] = False
video_capture_dict = {}


@app.route('/objectdetection', methods=['GET'])
def fuck():
    return "Please use POST method for object detection"


@app.route('/objectdetection', methods=['POST'])
def predict():
    request_json = request.json
    img = cv2.imdecode(np.frombuffer(request.data, np.uint8), cv2.IMREAD_UNCHANGED)
    height, width, _ = img.shape
    rows = img.shape[0]
    cols = img.shape[1]
    t1 = time.time()
    cvNet.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
    cvOut = cvNet.forward()
    print(time.time()-t1)
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
    print(output_result_list)
    return jsonify(output_result_list)

#
# if __name__ == "__main__":
#     app.run(host='0.0.0.0',threaded=True)
