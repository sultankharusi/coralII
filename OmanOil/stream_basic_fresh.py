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
parser.add_argument(
    '--camera_idx', help='Index of which video source to use. ', default="0")
args = parser.parse_args()
# rtsp://admin:AH@198712@192.168.1.88:554/Streaming/channels/2/
#rtsp = "rtsp://admin:Hik12345@192.168.1.64:554/Streaming/channels/1/"
#rtsp = "http://admin:Hik12345@192.168.1.64:445/doc/page/login.asp?_1660042995178"
cam_id = args.camera_idx

if cam_id.isnumeric():
    cam_id = int(cam_id)

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
    cap.set(cv.CAP_PROP_FPS, 16)
    cap = FreshestFrame(cap)
    ret = 0
    while True:
        ret, frame = cap.read(seqnumber=ret+1)
        if not ret:
            break
        cv2_im = frame[320:720, 265:1650]
        #cv2.imwrite(f"frames/{frmz}.jpg", cv2_im)
        frmz += 1
        #cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        #cv2_im = cv.resize(cv2_im, (640, 640))
        #run_inference(interpreter, cv2_im_rgb.tobytes())
        #objs = get_objects(interpreter, args.threshold)[:args.top_k]
        cv.imshow('frame', cv2_im)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    print("frmz", frmz)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()