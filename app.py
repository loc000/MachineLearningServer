import csv
import time

import keras
from PIL import Image

filter_class_label_file = "filter_class.csv"
save_debug_output = True
enable_image_caption = True
port = 5000
import os
import socket

import cv2
import flask
import numpy as np
from DenseDepth import test2

from flask import jsonify
from flask import request
from zeroconf import ServiceInfo, Zeroconf, get_all_addresses
from waitress import serve

cwd = os.getcwd()
import tensorflow.compat.v1 as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

tf.disable_v2_behavior()
if enable_image_caption:
    # import tensorflow.compat.v1 as tf
    # tf.disable_v2_behavior()
    os.chdir("./ShowAttendAndTellModel/")
    print(os.getcwd())
    from ShowAttendAndTellModel.run_inference import CaptionInference

    os.chdir(cwd)

os.chdir("./ssd_mobilenet_v2_oid_v4_2018_12_12/")
model_weight_file = 'frozen_inference_graph.pb'
model_graph_file = 'graph.pbtxt'
if not os.path.isfile(model_weight_file):
    import wget

    wget.download(
        "https://github.com/loc000/MachineLearningServer/releases/download/ssd_mobilenet_v2_oid_v4_2018_12_12/frozen_inference_graph.pb",
        out=model_weight_file)
    # wget.download("https://github.com/loc000/MachineLearningServer/releases/download/ssd_mobilenet_v2_oid_v4_2018_12_12/graph.pbtxt",out=model_graph_file)
cvNet = cv2.dnn.readNetFromTensorflow(model_weight_file, model_graph_file)
# cvNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
classList = [line.rstrip('\n') for line in open("oid_v4_label_map.txt")]
os.chdir(cwd)

app = flask.Flask(__name__)
app.config["DEBUG"] = False
video_capture_dict = {}

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("192.168.43.1", 80))
except:
    s.connect(("192.168.0.1", 80))
my_ip = s.getsockname()[0]

frame_no = 0
filter_class = list(csv.reader(open(filter_class_label_file, encoding="utf-8"), delimiter=','))


@app.route('/objectdetection', methods=['POST'])
def predict():
    global frame_no
    global depth2
    frame_no = frame_no + 1
    request_json = request.json
    img = cv2.imdecode(np.frombuffer(request.data, np.uint8), cv2.IMREAD_UNCHANGED)
    height, width, _ = img.shape
    t1 = time.time()
    depth = np.asarray(cv2.resize(test2.predict(img), (width, height)))
    print("depth estimation inference time: ",time.time()-t1)
    rows = img.shape[0]
    cols = img.shape[1]
    cvNet.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
    cvOut = cvNet.forward()
    # Put efficiency information.
    t, _ = cvNet.getPerfProfile()
    print('Object detection inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency()))
    output_result_list = []
    if save_debug_output:
        display_img = img.copy()
    for key,detection in enumerate(cvOut[0, 0, :, :]):
        score = float(detection[2])
        className = classList[int(detection[1]) - 1]
        label_list = filter_class[int(detection[1])]
        img_write = img.copy()
        if int(label_list[2]) > 0 and score > 0.3:
            if len(label_list) >= 4 and label_list[3] and score < float(label_list[3]):
                print("skipped, {}, score={}".format(label_list[0],score))
                continue
            left = max(detection[3] * cols, 0)
            top = max(detection[4] * rows, 0)
            right = detection[5] * cols
            bottom = detection[6] * rows


            mid_x, mid_y = left + (right - left) / 2, top + (bottom - top) / 2
            loc = ""
            depth2 = np.average(depth[int(top):int(bottom), int(left):int(right)])
            priority = int(label_list[2]) * int(depth2)
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

            if save_debug_output:
                cv2.rectangle(img_write, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
                cv2.rectangle(display_img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210),
                              thickness=2)

                # print(classList[int(detection[1]) - 1])
                img_display_text = label_list[0]+" d="+str(depth2)
                cv2.putText(img_write, img_display_text, (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(display_img, img_display_text, (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 1, cv2.LINE_AA)
            output_result_list.append({"className": className,
                                       "left": (int(left)),
                                       "top": (int(top)),
                                       "right": int(right),
                                       "bottom": int(bottom),
                                       "location_description": loc,
                                       "depth": depth2,
                                       "priority": priority
                                       })
            if save_debug_output:
                output_object_folder = "output/byObject/" + className + label_list[1] + "/"
                if not os.path.isdir(output_object_folder):
                    os.makedirs(output_object_folder, exist_ok=True)
                # output_frame_folder = "output/byFrame/"
                # if not os.path.isdir(output_frame_folder):
                #     os.makedirs(output_frame_folder, exist_ok=True)
                cv2.imwrite(
                    output_object_folder + "frameNo_" + str(frame_no).zfill(6) +"_key_"+str(key)+ ".jpg",
                    img_write)
                # cv2.imwrite(output_frame_folder + "frameNo_" + str(frame_no).zfill(6) + ".jpg", display_img)
    # data["result"] = output_result_list
    if save_debug_output:
        output_frame_folder = "output/byFrame/"
        if not os.path.isdir(output_frame_folder):
            os.makedirs(output_frame_folder, exist_ok=True)
        cv2.imwrite(output_frame_folder + "frameNo_" + str(frame_no).zfill(6) + ".jpg", display_img)

        # cv2.imshow("depth",depth)
        cv2.imshow("result", display_img)
        cv2.waitKey(1)
        pass
    output_result_list = sorted(output_result_list, key = lambda i: i['priority'], reverse=True)
    print(output_result_list)

    return jsonify(output_result_list)


if enable_image_caption:
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
        t1 = time.time()
        alphas, betas, captions = cap_infer.inference_np(np.array([img_np]))
        print("Image caption inference time:,",time.time()-t1)
        # output_result_list = []
        out_caption = None
        for alpha, beta, caption in zip(alphas, betas, captions):
            out_caption = caption
            # out_file.write("{}\t{}\n".format(fname, caption))
            # if args.visualize:
            #     visualize(alpha, beta, caption, fname, args.use_inception)
            # output_result_list.append({"caption": caption})
        # data["result"] = output_result_list
        # print(output_result_list)
        return jsonify({"caption": out_caption})


if __name__ == "__main__":
    sess = tf.Session()
    keras.backend.set_session(sess)
    test2.set_session(sess)
    if enable_image_caption:
        sess = tf.Session()
        cap_infer = CaptionInference(sess, "./model_best/model-best", use_inception=True)
        # image_caption_info = ServiceInfo("_icml._tcp.local.",
        #                                  "_icml._tcp.local.",
        #                                  socket.inet_aton(my_ip), port, 0, 0,
        #                                  {'type': 'imagecaption'},
        #                                  "image_caption_machine_learning_server.tcp.local.")
        image_caption_info = ServiceInfo("_icml._tcp.local.",
                                         "_icml._tcp.local.", addresses=[socket.inet_aton(my_ip)],
                                         port=port,
                                         properties={'type': 'imagecaption'},
                                         server="image_caption_machine_learning_server.tcp.local.")

    # object_detection_info = ServiceInfo("_oml._tcp.local.",
    #                                     "_oml._tcp.local.",
    #                                     socket.inet_aton(my_ip), port, 0, 0,
    #                                     {'type': 'objectdetection'},
    #                                     "object_detection_machine_learning_server.tcp.local.")
    object_detection_info = ServiceInfo("_oml._tcp.local.",
                                        "_oml._tcp.local.", addresses=[socket.inet_aton(my_ip)],
                                        port=port,
                                        properties={'type': 'objectdetection'},
                                        server="object_detection_machine_learning_server.tcp.local.")

    zeroconf = Zeroconf()
    zeroconf.register_service(object_detection_info)
    if enable_image_caption:
        zeroconf.register_service(image_caption_info)
    else:
        print("image caption is disabled.")
    print("Broadcasting mdns on {}".format(my_ip))
    # app.run(host='0.0.0.0', port=port)
    serve(app, host='0.0.0.0', port=port)
