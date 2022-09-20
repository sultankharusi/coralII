import socket
import tqdm
import os
import threading
# device's IP address
SERVER_HOST = ""
SERVER_PORT = 201
# receive 4096 bytes each time
BUFFER_SIZE = 4096
SEPARATOR = "<SEPARATOR>"

def handle_client(client_socket, address):
    # receive the file infos
    # receive using client socket, not server socket
    connected = True

    while connected:

        received = client_socket.recv(BUFFER_SIZE).decode()
        filename, filesize = received.split(SEPARATOR)
        # remove absolute path if there is
        filename = os.path.basename(filename)
        # convert to integer
        filesize = int(filesize)

        # start receiving the file from the socket
        # and writing to the file stream
        progress = tqdm.tqdm(range(filesize), "Receiving {}".format(filename), unit="B", unit_scale=True, unit_divisor=1024)
        with open(filename, "wb") as f:
            while True:
                # read 1024 bytes from the socket (receive)
                bytes_read = client_socket.recv(BUFFER_SIZE)
                if not bytes_read:    
                    # nothing is received
                    # file transmitting is done
                    break
                # write to the file the bytes we just received
                f.write(bytes_read)
                # update the progress bar
                progress.update(len(bytes_read))
        connected = False


def start():
    # create the server socket
    # TCP socket
    s = socket.socket()

    # bind the socket to our local address
    s.bind((SERVER_HOST, SERVER_PORT))

    # enabling our server to accept connections
    # 5 here is the number of unaccepted connections that
    # the system will allow before refusing new connections
    s.listen(5)
    print("[*] Listening as {}:{}".format(SERVER_HOST, SERVER_PORT))

    while True:
        # accept connection if there is any
        client_socket, address = s.accept() 
        # if below code is executed, that means the sender is connected
        print("[+] {} is connected.".format(address))
        thread = threading.Thread(target=handle_client, args=(client_socket, address))
        thread.start()
        print("[***][ACTIVE CONNECTIONS] {}".format(threading.activeCount() -1))
start()

# close the client socket
client_socket.close()
# close the server socket
s.close()