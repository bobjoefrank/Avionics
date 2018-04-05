import subprocess
import os
import fcntl
import tempfile
import threading
from multiprocessing import Process, Value
import binascii
import time

from python_ocr import *
# import python_cnn as cnn

manager = Value('i', 0)

def printshit(fifo, model,mapping):
    fd = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)     # open the fifo, readonly, dont block on open
    oflags = fcntl.fcntl(fd, fcntl.F_GETFL)             # get any flags on it
    fcntl.fcntl(fd, fcntl.F_SETFL, oflags | os.O_NONBLOCK)  # add non-block
    wait_new_file = True
    images = []
    img_buff = bytearray()
    bytes_remaining = 0
    while manager.value != 1:
        BYTES = 10
        # d = bytes()
        try:
            d = os.read(fd, BYTES)
        except BlockingIOError:
            pass
        else:
            if d:                           # if we have new data
                if wait_new_file:           # figure out how many bytes we're reading
                    wait_new_file = False
                    size_seq = bytearray()
                    img_buff.extend(d)

                    for i in range(len(img_buff)):
                        byte = img_buff[i]
                        if byte != 0:
                            size_seq.append(byte)
                        else:               # hit the \0 end of string sequence for # of bytes
                            img_buff = img_buff[i+1:]   # extend the remaining data because its part of the image
                            break
                    # print(size_seq)
                    bytes_remaining = int(bytes(size_seq).split(b'\0',1)[0].decode())
                    print("new / bytes:", bytes_remaining)
                    # print("curr buff:", img_buff)
                else:
                    # new image
                    img_buff.extend(d)      # append new bytes to the buffer
                    if len(img_buff) >= bytes_remaining:    # we have all the bytes
                        image = img_buff[:bytes_remaining]  # take the ones we need
                        images.append(image)                # make the image
                        # print('img!')
                        # print(image)
                        # print()
                        # with open(str(len(images)) + '.jpg', 'wb') as f2:
                        #     f2.write(image)
                        wait_new_file = True
                        img_buff = img_buff[bytes_remaining:]   # carry over to next iteration
    os.close(fd)
    for image in images:
        test(image,model,mapping)
        # cnn.run_model(image)
        # ps = threading.Thread(target=test, args=(image,model,mapping))
        # ps.start()


def main(inp_image):

    #model and weights location
    bin_dir = '../models'

    #load model
    model = load_model(bin_dir)
    model._make_predict_function()
    mapping = pickle.load(open('%s/mapping.p' % bin_dir, 'rb'))

    tmpdir = tempfile.mkdtemp()
    filename = os.path.join(tmpdir, 'fifo')
    os.mkfifo(filename)
    print(filename)
    ps = threading.Thread(target=printshit, args=(filename,model,mapping))
    ps.start()
    proc = subprocess.Popen(['./OGdetect', inp_image, filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        outs, err = proc.communicate()
        # print(outs.decode())
        # with open('test.jpg', 'w+') as f:
        #     pass
            # f.write(outs)
    except Exception as e:
        print(e)
    manager.value = 1
    ps.join()
if __name__ == '__main__':
    main('../pictures/200zedit.jpg')
