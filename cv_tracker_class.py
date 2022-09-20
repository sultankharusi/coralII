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

class vision:
    def __init__(self, model_path, label_path, threshold, camera_id, target_filter):
        self.model = model_path
        self.label = label_path
        self.threshold = threshold
        self.target_filter = target_filter
        self.top_k= 22

        self.camera_id = camera_id

        print('Loading {} with {} labels.'.format(self.model, self.label))

    def initial_sequence(self):
        self.interpreter = make_interpreter(self.model)
        self.interpreter.allocate_tensors()
        self.labels = read_label_file(self.label)
        self.inference_size = input_size(self.interpreter)

    def start_cv(self):

        cap = cv2.VideoCapture(self.camera_id)
        tracker = Sort()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2_im = frame

            cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            cv2_im_rgb = cv2.resize(cv2_im_rgb, self.inference_size)
            run_inference(self.interpreter, cv2_im_rgb.tobytes())
            objs = get_objects(self.interpreter, self.threshold)[:self.top_k]
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
            cv2_im = self.append_objs_to_img(cv2_im, self.inference_size, treks, self.labels)
            cv2.imshow('frame', cv2_im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def append_objs_to_img(self, cv2_im, inference_size, objs, labels):
        # Add cv2_im as a global variable instead of eating memory being passed around with new pointers
        height, width, channels = cv2_im.shape
        scale_x, scale_y = width / inference_size[0], height / inference_size[1] # This is redundent and repeats on every frame
        for obj in objs:
            x0, y0, x1, y1 = int(scale_x*obj[0]), int(scale_y*obj[1]), int(scale_x*obj[2]), int(scale_y*obj[3])

            cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2_im = cv2.putText(cv2_im, str(obj[4]), (x0, y0+30),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        return cv2_im