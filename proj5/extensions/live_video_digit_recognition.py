import cv2 as cv
import torch
import numpy as np
from task1.model import MyNetwork
import sys


def live_video_digit_recognition(network):
    """
    For each frame, preprocess it and feed into the pretrained digit recognition model to make the prediction.
    :param network: the pretrained model
    """
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
        # threshold the grayscale image with 100 as the threshold, also inverse the intensity
        _, gray = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV)

        img = np.array(gray).astype('float32')
        # expand the dimension to meet the requirement of the model input
        img = np.expand_dims(img, [0, 1])
        tensor = torch.from_numpy(img)
        output = network(tensor)
        prediction = output.data.max(1, keepdim=True)[1].item()

        # Display the resulting frame
        print(f"Prediction: {prediction}")
        cv.imshow('digit', gray)
        key_press = cv.waitKey(1)
        if key_press == ord('q'):
            break
        if key_press == ord('s'):
            cv.imwrite('./screenshot.jpg', gray)
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


def main(argv):
    """
    Load the network and call the live video digit recognition function
    :param argv: command line parameters
    """
    network = MyNetwork()
    network.load_state_dict(torch.load('../task1/results/model.pth'))
    live_video_digit_recognition(network)


if __name__ == "__main__":
    main(sys.argv)
