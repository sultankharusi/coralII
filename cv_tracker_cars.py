import argparse
import cv2
import os
import numpy as np

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference
from sort import *

def main():
    default_model_dir = ''
    default_model = 'tf2_mobilenet2_tpu.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 1)
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='classifier score threshold')
    parser.add_argument('--tools_file', help='Indicies of the frame to be streamed')
    args = parser.parse_args()

    ROI = []
    if args.tools_file:

        if args.tools_file.endswith('json'):
            import json
            with open(args.tools_file) as json_file:
                tools = json.load(json_file)
            if 'ROI' in tools.keys():
                ROI = list(tools['ROI'].values())


    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)

    cap = cv2.VideoCapture(args.camera_idx)
    tracker = Sort()
    while cap.isOpened():
        ret, cv2_im = cap.read()
        if not ret:
            print("INference Size: ", inference_size)
            break

        if ROI:
            cv2_im = cv2_im[ROI[2]:ROI[3],ROI[0]:ROI[1]]

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        run_inference(interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(interpreter, args.threshold)[:args.top_k]
        if objs:
            # x1,y1,x2,y2, score
            buf_list = []
            for i in objs:
                ll = list(i[2])
                ll.append(i[1])
                buf_list.append(ll)
            np_array = np.array(buf_list) # x1,y1,x2,y2, score
            treks = tracker.update(np_array)

        else:
            treks = tracker.update()
        print("Treks", treks)
        print("objects", objs)
        cv2_im = append_objs_to_img(cv2_im, inference_size, treks, labels)
        cv2.imshow('frame', cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        x0, y0, x1, y1 = int(scale_x*obj[0]), int(scale_y*obj[1]), int(scale_x*obj[2]), int(scale_y*obj[3])

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, str(obj[4]), (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im

if __name__ == '__main__':
    main()