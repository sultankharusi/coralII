from __future__ import print_function
from enum import unique
import sys
import threading


import argparse
import cv2
import os
import numpy as np
from collections import OrderedDict
import time
import requests
import random
import json

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference


#os.chdir("home/mendel/repo/coralII/OmanOil")
os.system("sudo ifconfig eth0 down")
os.system("sudo ifconfig eth0 169.254.0.211")
os.system("sudo ifconfig eth0 up")

from sort import *


class FixSizeOrderedDict(OrderedDict):
    def __init__(self, *args, max=0, **kwargs):
        self._max = max
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        if self._max > 0:
            if len(self) > self._max:
                #print("Popped! ")
                self.popitem(False)

def addObject(my_dictionary, id):
        nor = dict({ k:None for k in ('Plate','scync','out_time')})
        my_dictionary[id] = nor
        return my_dictionary
    

# API stuff
def send_file(data_raw, file_name, image=True, json=False):
    add_image_url = "https://ai-maestro-demo.com/fastapi-db/AddVehicleImages"
    if image:
        cv2.imwrite(file_name, data_raw)
        files = [('images', (file_name,open(file_name, 'rb'), 'image/jpg'))]
    if json:
        with open(file_name, "w") as write_file:
            json.dump(data_raw, write_file, indent=4)
    upload_status = requests.post(add_image_url, files=files)
    os.remove(file_name)



import time 
def get_time(date=False, simple=False):
    t = time.localtime()
    if date:
        return time.strftime("%Y-%m-%d", t)
    if simple:
        return time.strftime("%H_%M_%S", t)   
    return time.strftime("%H:%M:%S", t)



def is_inside(center, box):
    if center[0] > box[0] and center[0] < box[2] and center[1] < box[3]: # Check box positions
        return True
    else:
        return False

def sync_object(vehicles, index_):
    vehicle = vehicles[index_]
    url = "https://ai-maestro-demo.com/fastapi-db/AddVehicle/"
    #print(vehicle["id"])
    unique_id = int(float(f"{vehicle['id']}{random.randint(1,999)}"))
    name = f"{unique_id}_{vehicle['pump']}_{vehicle['side']}_{vehicle['plate']}.jpg"
    #print('sending info...', vehicle) # Update info
    V_img = get_scaled_box(vehicle["box"])
    send_file(V_img, name)
    time = vehicle['entry_time']
    data = [
        {"id":unique_id,
        "project_name":"omanoil",
        "pump":vehicle['pump'],
        "side":vehicle['side'],
        "plate":vehicle['plate'],
        "image_name":name,
        "date": get_time(date=True),
        "entry_time": time,
        "exit_time": time,
        "stay_time": '00:00:00',
        }]
    posting = requests.post(url, json = data)
    vehicles[index_]['sync'] = True
    return vehicles

def box_centeres_match(plate_centers, vehicle_box):
    keys_ = vehicle_box.keys()
    for center in plate_centers.keys():
        for i in keys_:
            if not vehicle_box[i]['plate']:
                box = vehicle_box[i]['box']
                #print(box)
                if is_inside(center, box):
                    plate = plate_inference(plate_centers[center[:3]], vehicle_box[i]) # Run inference on plate location only print this!! write this functiona and make the frame global... Do not foget the scale thingy
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
#print('Loading {} with {} labels.'.format(case_model, case_model_labels))
case_interpreter = make_interpreter(case_model)
case_interpreter.allocate_tensors()
case_labels = read_label_file(case_model_labels)
case_inference_size = input_size(case_interpreter)

def char_align(resutls):
    pairs = {}
    for i in resutls:
        pairs[i[2][0]] = i[0]
    return OrderedDict(sorted(pairs.items()))

def get_scaled_box(box, yscale=0.96, xscale=0.256, box_only = False):
    y1,y2,x1,x2 = int(box[1]/yscale),int(box[3]/yscale),int(box[0]/xscale),int(box[2]/xscale)
    if box_only:
        return y1,y2,x1,x2
    return cv2_im_cropped[y1:y2,x1:x2]

def plate_inference(plate,V_box,yscale=0.96,xscale=0.256): # OmanOil yscale 1.22, xscale 0.36 # Video is yscale=0.96,xscale=0.256
    #print("Plate_inference")
    #y1,y2,x1,x2 = int(plate[1]/yscale),int(plate[3]/yscale),int(plate[0]/xscale),int(plate[2]/xscale)
    frame = get_scaled_box(plate) #cv2_im_cropped[y1:y2,x1:x2]
    #cv2.imwrite("/home/mendel/repo/Plate_Cropped.jpg", frame)
    sq_frame = square_plates(frame)
    frame = cv2.resize(sq_frame, case_inference_size)
    #cv2.imwrite("/home/mendel/repo/Plate1.jpg", frame)
    run_inference(case_interpreter, frame.tobytes())
    objs = get_objects(case_interpreter, 0.5)[:6]
    if objs:
        plate_clz = char_align(objs)
        plate_final = [case_labels[i] for i in plate_clz.values()]
        plate_final = ''.join(plate_final)
        print("plate # recognized!..", plate_final)
        pl_name = plate_final+'_'+get_time(simple=True)+'_'+V_box["side"]+".jpg"
        #print(pl_name, type(sq_frame))
        send_file(sq_frame, pl_name)
        return plate_final
    else:
        return None

def delete_sequence(tracks_status, i):
    tracks_status[i]["exit_time"] = get_time()
    tracks_status[i]["missing_frames"] += 1
    
    if not tracks_status[i]["missing_frames"]%10:
        Update_url = "https://ai-maestro-demo.com/fastapi-db/UpdateVehicle/"
        update_status = requests.post(Update_url, json={"id":tracks_status[i]["id"], "exit_time":tracks_status[i]["exit_time"]})
        #print("outtime")
    elif tracks_status[i]["missing_frames"] > 50:
        popped = tracks_status.pop(i)
        #print("deleted !!", popped)    
    return tracks_status

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
    #print(tracks_status, "1")
    #print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)
    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap = FreshestFrame(cap)
    ret = 0
    tracker = Sort(50,50)
    pump = 6
    
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
                if i[0] in filtered and i[2][3] >= 270:    
                    ll = list(i[2]) # Declare a list and append the bounding box x1y1x2y2
                    #print("Car center,", i[2])
                    ll.append(i[1]) # Append the score
                    buf_list.append(ll) # Append as a whole list
                else:
                    ll = list(i[2]) # Declare a list and append the bounding box x1y1x2y2
                    ll.append(i[1]) # Append the score
                    plate_list[get_center(i[2])] = ll # Append as a whole list with the center being the key
                    #print("Plate Center! ", get_center(i[2]), i[2])
                    
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
                 # Go through delete sequence ... Set end time and all
                tracks_status = delete_sequence(tracks_status, i)
        #print(tracks_status, "4")
        for i in treks:
            side = None
            if i[4] not in tracks_status.keys():
                if i[:4][2] > 190:
                    side = "A"
                else:
                    side = "B"
                tracks_status[i[4]] = dict({'id':i[4],'box':i[:4],'plate':None,
                                            'entry_time': get_time(), "pump":pump,"side":side,'sync':None,
                                            'exit_time':None, "missing_frames":0})
            else:
                tracks_status[i[4]]["missing_frames"] = 0
                print("Reset missing frames!")
                
        #print(tracks_status, "5")
        if tracks_status and plate_list: # Perfect
            #print("Matching centers!")
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
