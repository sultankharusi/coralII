from __future__ import print_function
import sys
import threading


import argparse
import cv2
import os
import numpy as np
from collections import OrderedDict
import time

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference
from sort import *


class FixSizeOrderedDict(OrderedDict):
    def __init__(self, *args, max=0, **kwargs):
        self._max = max
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        if self._max > 0:
            if len(self) > self._max:
                print("Popped! ")
                self.popitem(False)

def addObject(my_dictionary, id):
        nor = dict({ k:None for k in ('Plate','scync','out_time')})
        my_dictionary[id] = nor
        return my_dictionary
    

import time 
def get_time():
    t = time.localtime()
    return time.strftime("%D:%H:%M:%S", t)



def is_inside(center, box):
    if center[0] > box[0] and center[0] < box[2] and center[1] < box[1]: # Check box positions
        return True
    else:
        return False

def sync_object(vehicles, index_):
    print('sending info...', vehicles[index_]) # Update info
    vehicles[index_]['sync'] = True
    return vehicles

def box_centeres_match(plate_centers, vehicle_box):
    keys_ = vehicle_box.keys()
    for center in plate_centers.keys():
        for i in keys_:
            if not vehicle_box[i]['plate']:
                box = vehicle_box[i]['box']
                if is_inside(center, box):
                    plate = plate_inference(plate_centers[center[:4]]) # Run inference on plate location only print this!! write this functiona and make the frame global... Do not foget the scale thingy
                    if plate:
                        vehicle_box[i]['plate'] = plate
                        vehicle_box = sync_object(vehicle_box, i)
    return vehicle_box
    
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

def square_plates(plate):
    H, W = plate.shape[0], plate.shape[1]
    if int(W*0.75) > H:
        padding = int(W*0.75) - H
        return cv2.copyMakeBorder(plate, 0, padding, 0, 0, cv2.BORDER_CONSTANT, None, (0,0,0))#, H+padding, W
    return plate#, H, W

def get_center(box):
        x0, y0, x1, y1 = box
        return (x0+x1)/2, (y0+y1)/2

case_model = "model_APNR_edgetpu.tflite"
case_model_labels = "label_OCR.txt"
print('Loading {} with {} labels.'.format(case_model, case_model_labels))
case_interpreter = make_interpreter(case_model)
case_interpreter.allocate_tensors()
case_labels = read_label_file(case_model_labels)
case_inference_size = input_size(case_interpreter)

def plate_inference(plate,yscale=1.22,xscale=0.36):
    y1,y2,x1,x2 = int(plate[1]/yscale),int(plate[3]/yscale),int(plate[0]/xscale),int(plate[2]/xscale)
    frame = cv2_im_cropped[int(plate[1]):int(plate[3]),int(plate[0]):int(plate[2])]
    frame = cv2.resize(square_plates(frame), case_inference_size)
    run_inference(case_interpreter, frame.tobytes())
    objs = get_objects(case_interpreter, 0.6)[:6]
    if objs:
        print("plate # recognized!..", objs)
        return objs
    else:
        return None

class FreshestFrame(threading.Thread):
    def __init__(self, capture, name='FreshestFrame'):
        self.capture = capture
        assert self.capture.isOpened()

        # this lets the read() method block until there's a new frame
        self.cond = threading.Condition()

        # this allows us to stop the thread gracefully
        self.running = False

        # keeping the newest frame around
        self.frame = None

        # passing a sequence number allows read() to NOT block
        # if the currently available one is exactly the one you ask for
        self.latestnum = 0

        # this is just for demo purposes        
        self.callback = None
        
        super().__init__(name=name)
        self.start()

    def start(self):
        self.running = True
        super().start()

    def release(self, timeout=None):
        self.running = False
        self.join(timeout=timeout)
        self.capture.release()

    def run(self):
        counter = 0
        while self.running:
            # block for fresh frame
            (rv, img) = self.capture.read()
            assert rv
            counter += 1

            # publish the frame
            with self.cond: # lock the condition for this operation
                self.frame = img if rv else None
                self.latestnum = counter
                self.cond.notify_all()

            if self.callback:
                self.callback(img)

    def read(self, wait=True, seqnumber=None, timeout=None):
        # with no arguments (wait=True), it always blocks for a fresh frame
        # with wait=False it returns the current frame immediately (polling)
        # with a seqnumber, it blocks until that frame is available (or no wait at all)
        # with timeout argument, may return an earlier frame;
        #   may even be (0,None) if nothing received yet

        with self.cond:
            if wait:
                if seqnumber is None:
                    seqnumber = self.latestnum+1
                if seqnumber < 1:
                    seqnumber = 1
                
                rv = self.cond.wait_for(lambda: self.latestnum >= seqnumber, timeout=timeout)
                if not rv:
                    return (self.latestnum, self.frame)

            return (self.latestnum, self.frame)


def main():
    global cv2_im_cropped
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
    parser.add_argument('--top_k', type=int, default=4,
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
    
    classes_of_interest = ['Vehicle']
    filtered = [clsz.index(i) for i in classes_of_interest]
    tracks_status = FixSizeOrderedDict(max=4)
    print(tracks_status, "1")
    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)
    
    

    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FPS, 35)
    cap = FreshestFrame(cap)
    ret = 0
    tracker = Sort()
    
    emptyslot = dict({ k:None for k in ('box','score','plate','intime','scync','out_time')})
    while True:
        ret, frame = cap.read(seqnumber=ret+1)
        if not ret:
            break
        cv2_im_cropped = frame#[190:504,235:1300] # This is a global variable and shall never be altered, so cv2_im_cropped is never augmented after this point

        #cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_cropped, inference_size)
        run_inference(interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(interpreter, args.threshold)[:args.top_k]
        if objs:
            # x1,y1,x2,y2, score
            buf_list = []
            plate_list = {}
            for i in objs:
                if i[0] in filtered:    
                    ll = list(i[2]) # Declare a list and append the bounding box x1y1x2y2
                    print("Car center,", i[2])
                    ll.append(i[1]) # Append the score
                    buf_list.append(ll) # Append as a whole list
                else:
                    ll = list(i[2]) # Declare a list and append the bounding box x1y1x2y2
                    ll.append(i[1]) # Append the score
                    plate_list[get_center(i[2])] = ll # Append as a whole list with the center being the key
                    print("Plate Center! ", get_center(i[2]))
                    
            np_array = np.array(buf_list) # x1,y1,x2,y2, score
            if buf_list:
                treks = tracker.update(np_array)
            else:
                treks = tracker.update()
        else:
            treks = tracker.update()
        
        #print("Treks", treks)
        #print("objects", objs)
        #print(tracks_status, "2")
        keys_ = list(tracks_status.keys())
        #print(tracks_status, "3")
        for i in keys_:
            if i not in [l[4] for l in treks]:
                tracks_status.pop(i) # Go through delete sequence ... Set end time and all
        #print(tracks_status, "4")
        for i in treks:
            if i[4] not in tracks_status.keys():
                tracks_status[i[4]] = dict({'box':i[:3],'plate':None,'intime': get_time(),'sync':None,'outtime':None})
        #print(tracks_status, "5")
        if tracks_status and plate_list:
            tracks_status = box_centeres_match(plate_list,tracks_status)
        
        cv2_im = append_objs_to_img(cv2_im_cropped, inference_size, treks, labels)
        #print(tracks_status, "6")
        cv2.imshow('frame', cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
