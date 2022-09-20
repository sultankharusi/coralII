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
    default_stream_model_dir = ''
    default_stream_model = 'Plates_vehicle_edgetpu.tflite'
    default_stream_labels = 'label_VP.txt'
    case_model = "model_APNR_edgetpu.tflite"
    case_model_labels = "label_OCR.txt"
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_stream_model_dir,default_stream_model))
    parser.add_argument('--case_model', help='.tflite case model path',
                        default=os.path.join(default_stream_model_dir,case_model))
    parser.add_argument('--case_labels', help='.tflite case model path',
                        default=os.path.join(default_stream_model_dir,case_model_labels))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_stream_model_dir, default_stream_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', help='Index of which video source to use. ', default = "1") # camera id 1 is because no 0 in the dev borad
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='classifier score threshold')

    args = parser.parse_args()

    cam_id = args.camera_idx
    if cam_id.isnumeric():
        cam_id = int(cam_id)
    
    with open(default_stream_labels , 'r') as f:
        clsz = [i.strip('\n') for i in f.readlines()]
    
    classes_of_interest = ['Vehicle',"Plate"]
    filtered = [clsz.index(i) for i in classes_of_interest]

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)
    
    print('Loading {} with {} labels.'.format(args.case_model, args.case_labels))
    case_interpreter = make_interpreter(args.case_model)
    case_interpreter.allocate_tensors()
    case_labels = read_label_file(args.case_labels)
    case_inference_size = input_size(case_interpreter)
    logger = open("logger_sort_latency.txt", 'w')

    cap = cv2.VideoCapture(cam_id)
    tracker = Sort(35, 50)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        run_inference(interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(interpreter, args.threshold)[:args.top_k]
        if objs:
            # x1,y1,x2,y2, score
            buf_list = []
            logger.write("Frame")
            logger.write("\n")
            for i in objs:
                logger.write(str(i))
                logger.write('\n')
                if i[0] in filtered:    
                    ll = list(i[2])
                    ll.append(i[1])
                    buf_list.append(ll)
            np_array = np.array(buf_list) # x1,y1,x2,y2, score
            if buf_list:
                treks = tracker.update(np_array)
            else:
                treks = tracker.update()
        else:
            treks = tracker.update()
        print("Treks", treks)
        print("objects", objs)
        cv2_im = append_objs_to_img(cv2_im, inference_size, treks, labels)
        cv2.imshow('frame', cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.close()
            break
    logger.close()
    cap.release()
    cv2.destroyAllWindows()

def get_patch_from_src(file_name, orig, point):
    y_scale, x_scale = (400/320), (1500/320)
    y_range, x_range = (320+int(y_scale*point[0][0]), 320+int(y_scale*point[1][0])), (250+int(x_scale*point[0][1]), 250+int(x_scale*point[1][1]))
    cv2.imwrite(file_name,orig[y_range[0]:y_range[1], x_range[0]:x_range[1]])

def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1] # This is redundent and repeats on every frame
    for obj in objs:
        x0, y0, x1, y1 = int(scale_x*obj[0]), int(scale_y*obj[1]), int(scale_x*obj[2]), int(scale_y*obj[3])

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, str(obj[4]), (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im

if __name__ == '__main__':
    main()