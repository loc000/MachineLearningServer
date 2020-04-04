import os
import time
import cv2
import flask
import numpy as np
from PIL import Image

# cwd = os.getcwd()

# os.chdir("./ShowAttendAndTellModel/")
# print(os.getcwd())

# os.chdir(cwd)

# os.chdir("./ssd_mobilenet_v2_oid_v4_2018_12_12/")
cvNet = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'graph.pbtxt')
classList = [line.rstrip('\n') for line in open("oid_v4_label_map.txt")]
# os.chdir(cwd)

cap = cv2.VideoCapture("Joker_movie_stairs_now_popular_tourist_draw.mp4")
_, img = cap.read()
height, width, _ = img.shape
out = cv2.VideoWriter('test.out.mp4',cv2.VideoWriter_fourcc(*"mp4v"), 30, (width,height))
while cap.isOpened():
    _, img = cap.read()
    try:
        height, width, _ = img.shape
    except:
            break
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
            cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
            # print(classList[int(detection[1]) - 1])
            cv2.putText(img, classList[int(detection[1]) - 1], (int(left), int(top+35)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 1, cv2.LINE_AA)

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

            # output_result_list.append({"className": classList[int(detection[1]) - 1],
            #                            "left": (int(left)),
            #                            "top": (int(top)),
            #                            "right": int(right),
            #                            "bottom": int(bottom),
            #                            "location_description": loc
            #                            })
    # data["result"] = output_result_list
    out.write(img)
    # cv2.imshow("loc",img)
    # cv2.waitKey(1)
out.release()
 