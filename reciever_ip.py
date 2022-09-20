import socket
import os
import threading

# device's IP address
SERVER_HOST = ""
SERVER_HOST = socket.gethostbyname(socket.gethostname())
SERVER_PORT = 202
# receive 4096 bytes each time
HEADER = 4096
FORMAT = "utf-8"
SEPARATOR = "<SEPARATOR>"
DISCONNECT_MESSAGE = "! DISCONNECT"

# create the server socket
# TCP socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# bind the socket to our local address
s.bind((SERVER_HOST, SERVER_PORT))

# enabling our server to accept connections
# 5 here is the number of unaccepted connections that
# the system will allow before refusing new connections

def handle_client(conn, addr):
    print("[NEW CONNECTION] {} connected".format(addr))

    connected = True
    while connected:
        msg_length = conn.recv(HEADER).decode(FORMAT) # This blocks the lines of code
        if msg_length:
            msg_length = int(msg_length)
            msg = conn.recv(msg_length).decode(FORMAT)
            print("{} {}".format(addr, msg))

            if msg == DISCONNECT_MESSAGE:
                connected = False
    conn.close()

def start():
    s.listen()
    print("[*] Listening as {}:{}".format(SERVER_HOST, SERVER_PORT))
    while True:
        conn, addr = s.accept() # This blocks the lines of code
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        print("[ACTIVE CONNECTIONS] {}".format(threading.activeCount() -1))

print("Firing up server !")
start()