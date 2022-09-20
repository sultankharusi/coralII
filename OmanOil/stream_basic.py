import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--camera_idx', help='Index of which video source to use. ', default = "0") # camera id 1 is because no 0 in the dev boradss
args = parser.parse_args()
#rtsp://admin:AH@198712@192.168.1.88:554/Streaming/channels/2/
rtsp = "rtsp://admin:Hik12345@192.168.1.64:554/Streaming/channels/1/"
#rtsp = "http://admin:Hik12345@192.168.1.64:445/doc/page/login.asp?_1660042995178"
cam_id = args.camera_idx

if cam_id.isnumeric():
	cam_id = int(cam_id)

def main():

	cap = cv2.VideoCapture(cam_id)
	while cap.isOpened():
	        ret, frame = cap.read()
	        if not ret:
	            break
	        cv2_im = frame

	        #cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
	        cv2_im = cv2.resize(cv2_im, (640,640))
	        #run_inference(interpreter, cv2_im_rgb.tobytes())
	        #objs = get_objects(interpreter, args.threshold)[:args.top_k]
	        cv2.imshow('frame', cv2_im)
	        if cv2.waitKey(1) & 0xFF == ord('q'):
	            break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()