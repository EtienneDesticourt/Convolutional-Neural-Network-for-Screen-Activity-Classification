from PIL import ImageGrab, Image
import os, threading, time

def captureScreen(path, index, thumbSize, quality):
    capture = ImageGrab.grab()
    capture.thumbnail(thumbSize, Image.ANTIALIAS)
    savePath = os.path.join(path, "image"+str(index)+".jpg")
    capture.save(savePath, optimize=True, quality=quality)

def runCapture(path, interval, size, quality):
    index = 0
    files = os.listdir(path)
    index += len(files)
    while 1:
        captureScreen(path, index, size, quality)
        time.sleep(interval)
        index += 1

PATH = r"C:\Users\Etienne\Desktop\Captures"
INTERVAL = 10
SIZE = (800, 800)
QUALITY = 50

MyThread = threading.Thread(target=runCapture,
                            args=(PATH, INTERVAL, SIZE, QUALITY))
MyThread.start()
