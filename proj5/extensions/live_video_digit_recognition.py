import cv2 as cv
import torch
import numpy as np
from task1.model import MyNetwork
import sys


def live_video_digit_recognition(network):
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    cv.namedWindow('digit', cv.WINDOW_NORMAL)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.resize(gray, (28, 28))

        img = np.array(gray).reshape((1, 1, 28, 28)).astype('float32')
        output = network(torch.Tensor(img))
        prediction = output.data.max(1, keepdim=True)[1].item()

        # Display the resulting frame
        print(f"Prediction: {prediction}")
        cv.imshow('digit', gray)
        if cv.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


def main(argv):
    network = MyNetwork()
    network.load_state_dict(torch.load('../task1/results/model.pth'))
    live_video_digit_recognition(network)
    return


if __name__ == "__main__":
    main(sys.argv)
