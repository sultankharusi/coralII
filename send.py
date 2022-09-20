import socket
import tqdm
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file', help='Index of which video source to use. ')
args = parser.parse_args()

def main():
	global net
	SEPARATOR = "<SEPARATOR>"
	host = "172.105.252.6"
	port = 201

	s = socket.socket()
	print(f"[+] Connecting to {host}:{port}")
	s.connect((host, port))

	print("[+] Connected.")
	filename = args.file
	BUFFER_SIZE = 4096 # send 4096 bytes each time step
	files_size = os.path.getsize(filename)
	file = "{}{}{}".format(filename, SEPARATOR,files_size)

	reply = s.send(file.encode())
	progress = tqdm.tqdm(range(files_size), f"Sending {filename}", unit="B", unit_scale=True, unit_divisor=1024)
	with open(filename, "rb") as f:
	    while True:
	        # read the bytes from the file
	        bytes_read = f.read(BUFFER_SIZE)
	        if not bytes_read:
	            # file transmitting is done
	            break
	        # we use sendall to assure transimission in 
	        # busy networks
	        s.sendall(bytes_read)
	        # update the progress bar
	        progress.update(len(bytes_read))


	print(reply)

if __name__ == "__main__":
    main()
