from __future__ import print_function
import sys
import threading

import cv2
import numpy as np
from collections import OrderedDict
import time
import requests
import os

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

def get_time(date=False, simple=False):
    t = time.localtime()
    if date:
        return time.strftime("%Y-%m-%d", t)
    if simple:
        return time.strftime("%H_%M_%S", t)   
    return time.strftime("%H:%M:%S", t)

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

def main():
    link = "rtsp://admin:omanoil7913@169.254.0.000/Streaming/channels/1/"
    cams = ["3","11","2","4","10","12","16","18","17","14","8","7","19","20"]
    cameras = [link.replace("000",i) for i in cams]
    #cameras = [0,1]
    for cam_id in cameras:
        cap = cv2.VideoCapture(cam_id)
        cap.set(cv2.CAP_PROP_FPS, 35)
        cap = FreshestFrame(cap)
        ret = 0
        while True:
            ret, frame = cap.read(seqnumber=ret+1)
            if not ret:
                break
            cv2_im_cropped = frame#[190:504,235:1300] # This is a global variable and shall never be altered, so cv2_im_cropped is never augmented after this point
            send_file(cv2_im_cropped, f"Craver_{get_time(simple=True)}.jpg")
            break
        cap.release()
        
if __name__ == '__main__':
    while True:
        main()
        time.sleep(300)