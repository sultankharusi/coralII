import socket
import tqdm
import os

SERVER_HOST = socket.gethostbyname(socket.gethostname())
SERVER_PORT = 5001
# receive 4096 bytes each time
HEADER = 4096
FORMAT = "utf-8"
SEPARATOR = "<SEPARATOR>"
DISCONNECT_MESSAGE = "! DISCONNECT"
SERVER = "192.168.100.12"
PORT = 5001


client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((SERVER, PORT))



