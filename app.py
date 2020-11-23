# -*- coding:utf-8 -*-
import cv2
import time
import json
import argparse
import numpy as np
from PIL import Image
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from load_model.pytorch_loader import load_pytorch_model, pytorch_inference
from flask import Flask,request,jsonify
import warnings
import uuid

# Ignore SourceChangeWarning
warnings.filterwarnings("ignore")

# initialize flask app
app = Flask(__name__)

# model = load_pytorch_model('models/face_mask_detection.pth');
model = load_pytorch_model('models/model360.pth');
# anchor configuration
#feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
feature_map_sizes = [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {1: 'NoMask', 0: 'Mask'}


def inference(image,
              conf_thresh=0.5,
              iou_thresh=0.4,
              target_shape=(360, 360),
              draw_result=True
              ):
    '''
    Main function of detection inference
    :param image: 3D numpy array of image
    :param conf_thresh: the min threshold of classification probabity.
    :param iou_thresh: the IOU threshold of NMS
    :param target_shape: the model input size.
    :param draw_result: whether to daw bounding box to the image.
    :param show_result: whether to display the image.
    :return:
    '''

    #copy = np.array(image)
    output_info = []
    height, width, _ = image.shape
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 250.0  # 归一化到0~1
    image_exp = np.expand_dims(image_np, axis=0)

    image_transposed = image_exp.transpose((0, 3, 1, 2))

    y_bboxes_output, y_cls_output = pytorch_inference(model, image_transposed)
    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=conf_thresh,
                                                 iou_thresh=iou_thresh,
                                                 )

    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)

        if draw_result:
            if class_id == 0:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
            #cv2.putText(image, "Body temperature: 36.6c", (xmin + 2, ymax + 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
            mask = bool(int(class_id) == 1)
            output_info.append([mask, conf, xmin, ymin, xmax, ymax])

    if draw_result:
        # save image and provide it in the api
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        im = Image.fromarray(image)
        id = uuid.uuid4()
        image_path = "predictions/%s.jpg" % id
        im.save(image_path)

        return output_info,image_path

    return output_info


'''
def run_on_video(video_path, output_video_name, conf_thresh):
    cap = cv2.VideoCapture(video_path)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # writer = cv2.VideoWriter(output_video_name, fourcc, int(fps), (int(width), int(height)))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not cap.isOpened():
        raise ValueError("Video open failed.")
        return
    status = True
    idx = 0
    while status:
        start_stamp = time.time()
        status, img_raw = cap.read()
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        read_frame_stamp = time.time()
        if (status):
            inference(img_raw,
                      conf_thresh,
                      iou_thresh=0.5,
                      target_shape=(360, 360),
                      draw_result=True,
                      show_result=False)
            cv2.imshow('image', img_raw[:, :, ::-1])
            cv2.waitKey(1)
            inference_stamp = time.time()
            # writer.write(img_raw)
            write_frame_stamp = time.time()
            idx += 1
            print("%d of %d" % (idx, total_frames))
            print("read_frame:%f, infer time:%f, write time:%f" % (read_frame_stamp - start_stamp,
                                                                   inference_stamp - read_frame_stamp,
                                                                   write_frame_stamp - inference_stamp))
    # writer.release()
'''

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        img = Image.open(request.files['file'])
        img = np.array(img)
        # get prediction
        prediction,image_path = inference(img, target_shape=(360, 360))
        profiles = []
        for face in prediction:
            face_square = str(face[2]),str(face[3]),str(face[4]),str(face[5])
            profile = {
                "mask": str(face[0]),
                "probability": str(face[1]),
                "face_square": face_square,
            }
            profiles.append(profile)
        return jsonify({"faces":profiles,"image":image_path})

'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Mask Detection")
    #parser.add_argument('--img-mode', type=int, default=1, help='set 1 to run on image, 0 to run on video.')
    parser.add_argument('--img-path', type=str, help='path to your image.')
    #parser.add_argument('--video-path', type=str, default='0', help='path to your video, `0` means to use camera.')
    #parser.add_argument('--hdf5', type=str, help='keras hdf5 file')
    args = parser.parse_args()
    #if args.img_mode:
    imgPath = args.img_path
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(inference(img, show_result=False, target_shape=(360, 360)))
    #else:
    #    video_path = args.video_path
    #    if args.video_path == '0':
    #        video_path = 0
    #    run_on_video(video_path, '', conf_thresh=0.5)
'''

if __name__ == '__main__':
    app.run()
