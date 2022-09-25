from __future__ import print_function

import argparse

import os
import sys
import time
import threading
import numpy as np
import cv2 as cv

parser = argparse.ArgumentParser()
# camera id 1 is because no 0 in the dev boradss
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
parser.add_argument('--threshold', type=float, default=0.4,
                    help='classifier score threshold')

args = parser.parse_args()
# rtsp://admin:AH@198712@192.168.1.88:554/Streaming/channels/2/
#rtsp = "rtsp://admin:Hik12345@192.168.1.64:554/Streaming/channels/1/"
#rtsp = "http://admin:Hik12345@192.168.1.64:445/doc/page/login.asp?_1660042995178"
cam_id = args.camera_idx

if cam_id.isnumeric():
    cam_id = int(cam_id)

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
tracker = Sort(35, 50)

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
    frmz = 0
    cap = cv.VideoCapture(cam_id)
    cap.set(cv.CAP_PROP_FPS, 35)
    cap = FreshestFrame(cap)
    ret = 0
    while True:
        ret, frame = cap.read(seqnumber=ret+1)
        if not ret:
            break
        cv2_im = frame#[320:720, 265:1650]
        #cv2.imwrite(f"frames/{frmz}.jpg", cv2_im)
        frmz += 1
        #cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        run_inference(interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(interpreter, args.threshold)[:args.top_k]
        if objs:
            # x1,y1,x2,y2, score
            buf_list = []
            for i in objs:
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
        cv.imshow('frame', cv2_im)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    print("frmz", frmz)
    cv.destroyAllWindows()
def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1] # This is redundent and repeats on every frame
    for obj in objs:
        x0, y0, x1, y1 = int(scale_x*obj[0]), int(scale_y*obj[1]), int(scale_x*obj[2]), int(scale_y*obj[3])

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, str(obj[4]), (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im

if __name__ == "__main__":
    main()