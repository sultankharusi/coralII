import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--camera_idx', help='Index of which video source to use. ', default = "0")
parser.add_argument('--tools_file', help='Indicies of the frame to be streamed')
args = parser.parse_args()
#rtsp://admin:AH@198712@192.168.1.88:554/Streaming/channels/2/
cam_id = args.camera_idx

ROI = []
if args.tools_file:

	if args.tools_file.endswith('json'):
		import json
		with open(args.tools_file) as json_file:
			tools = json.load(json_file)
		if 'ROI' in tools.keys():
			ROI = list(tools['ROI'].values())


			
if cam_id.isnumeric():
	cam_id = int(cam_id)
# variables
xi = -1
yi = -1
drawing = False

def draw(event, x, y, flag, param):
	global xi, yi, drawing
	"""global ix, iy, drawing, cv2_im
				if event == cv2.EVENT_LBUTTONDOWN:
					drawing = True
					ix = x
					iy = y
					print("Pressed")
			
				elif event == cv2.EVENT_MOUSEMOVE:
					if drawing == True:
						cv2.rectangle(cv2_im, pt1=(ix,iy), pt2=(x, y),color=(0,255,255),thickness=-1)
						print("Moving")
				elif event == cv2.EVENT_LBUTTONUP:
					drawing = False
					cv2.rectangle(cv2_im, pt1=(ix,iy), pt2=(x, y),color=(0,255,255),thickness=-1)
					cv2_im = cv2_im"""
	if event == cv2.EVENT_LBUTTONDOWN:
		drawing = True
		xi = x
		yi = y
		print(drawing)
	

def main():
	global cv2_im, drawing, xi, yi

	cap = cv2.VideoCapture(cam_id)
	cv2.namedWindow('frame')
	cv2.setMouseCallback('frame', draw)
	while cap.isOpened():
	        ret, cv2_im = cap.read()
	        if not ret:
	            break
	        if ROI:
	        	cv2_im = cv2_im[ROI[2]:ROI[3],ROI[0]:ROI[1]]
	        
	        #cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
	        #cv2_im = cv2.resize(frame, (640,640))
	        #run_inference(interpreter, cv2_im_rgb.tobytes())
	        #objs = get_objects(interpreter, args.threshold)[:args.top_k]
	        #cv2.circle(cv2_im, (171,158), 10, (0,255,0), 2)
	        cv2.circle(cv2_im, (xi,yi), 10, (0,255,0), 2)
	        cv2.putText(cv2_im, (f"{xi} Y {yi}"), (10,22),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,10,10),3)	
	        cv2.imshow('frame', cv2_im)
	        
	        if cv2.waitKey(1) & 0xFF == ord('s'):
	            break
	        elif cv2.waitKey(1) & 0xFF == ord('q'):
	        	print("Yawz####")
	        	cv2.imwrite("snapshot.jpg", cv2_im)

	        elif cv2.waitKey(1) & 0xFF == ord('b'):
	        	print("drawing")
	        	
	        	print(xi, yi)

	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()