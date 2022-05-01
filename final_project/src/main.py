import cv2 as cv
import os
import sys
import csv

os.chdir("../config")


def analyze():
    """
    Read these three csv files and calculate the accuracy
    """
    result_path = '../results'
    file_names = ['opencv.csv', 'self.csv', 'wechat.csv']
    for name in os.listdir(result_path):
        if name in file_names:
            path = os.path.join(result_path, name)
            with open(path, mode='r') as f:
                reader = csv.DictReader(f)
                result = next(reader)
                method = result["algorithm"]
                # delete the algorithm key-value pair
                del result["algorithm"]
                if '' in result:
                    del result['']
                correct_cnt = 0
                for k, v in result.items():
                    if v == '1':
                        correct_cnt += 1
                print(result)
                print(f'Accuracy for {method}: {correct_cnt / len(result)}')


def draw_bounding_box(mat, box):
    """
    Draw the bbox onto the mat based on a set of points
    :param mat: the image mat
    :param box: np.ndarray representing a bbox
    """
    for step in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        start = box[step[0]][0], box[step[0]][1]
        end = box[step[1]][0], box[step[1]][1]
        cv.line(mat, start, end, (0, 255, 0), 2)


class QRCodeDetector:
    # The QR code detector class, contains functions for single image detection, dataset evaluation and video detection.
    def __init__(self, mode, detector_name, detector):
        """
        Initialization function for this class
        :param mode: could be one of eval, video and image
        :param detector_name: could be opencv or wechat, used for file naming
        :param detector: the detector object
        """
        self.mode = mode
        self.detector_name = detector_name
        self.detector = detector

    def image_detect_segment(self, img_name):
        """
        Given an image name, detect and segment the QR code in that image mat
        :param img_name: the name of the image file
        """
        img_dir = f'../images/{img_name}.png'
        # if no such file, exit with code -1
        if not os.path.exists(img_dir):
            print("No such image in images folder!")
            exit(-1)

        img = cv.imread(img_dir)
        # call the detectAndDecode() function to get the contents and a set of bbox points
        contents, points, *malicious = self.detector.detectAndDecode(img)
        # draw the bbox onto the image mat if there is
        if len(contents) > 0:
            for box in points:
                draw_bounding_box(img, box.astype(int))

        while True:
            cv.imshow(img_name, img)
            keypress = cv.waitKey(1)
            if keypress == ord('q'):
                break
            if keypress == ord('s'):
                cv.imwrite(f"../results/{self.mode}_{self.detector_name}_{img_name}.png", img)
                print(f'This image is saved as ../results/{self.mode}_{self.detector_name}_{img_name}.png!')

    def images_evaluation(self):
        """
        Evaluate the detector's performance by applying it to the whole dataset in images folder.
        """
        src_folder = '../images'
        dst_folder = f'../results/{self.detector_name}_eval/'
        csv_dir = f'../results/{self.detector_name}.csv'
        if not os.path.exists(dst_folder):
            os.mkdir(dst_folder)
        results = {}
        for img_name in os.listdir(src_folder):
            img_path = os.path.join(src_folder, img_name)
            img = cv.imread(img_path)
            contents, points, *malicious = self.detector.detectAndDecode(img)
            if len(contents) > 0:
                results[img_name[:-4]] = 1
                for box in points:
                    draw_bounding_box(img, box.astype(int))
            else:
                results[img_name[:-4]] = 0
            cv.imwrite(f'{dst_folder}/{self.mode}_{self.detector_name}_{img_name}', img)
        print(results)

        # write the result into a csv file
        keys = list(results.keys())
        keys.insert(0, 'algorithm')
        values = list(results.values())
        values.insert(0, self.detector_name)
        with open(csv_dir, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(keys)
            w.writerow(values)

    def video_detect_segment(self):
        """
        Detect and segment QR code in a real time video
        """
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            contents, points, *malicious = self.detector.detectAndDecode(frame)
            if len(contents) > 0:
                for box in points:
                    draw_bounding_box(frame, box.astype(int))

            # Display the resulting frame
            cv.imshow('frame', frame)
            keypress = cv.waitKey(1)
            if keypress == ord('q'):
                break
            if keypress == ord('s'):
                cv.imwrite(f'../results/{self.mode}_{self.detector_name}.png', frame)
                print(f'This image is saved as ../results/{self.mode}_{self.detector_name}.png!')
        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()


def main(argv):
    """
    Construct the detector based on the method (opencv or wechat), call the corresponding function based on the mode
    (image, eval, video or analyze).
    :rtype: object return after analyzing
    """
    print(
        'Usage: python main.py opencv/wechat video || python main.py opencv/wechat image <image_name> || python main.py'
        'opencv/wechat eval || python main.py analyze')
    detector = None

    if argv is None or len(argv) > 4:
        print("Invalid parameters!")
        exit(-1)

    if argv[1] == 'opencv':
        detector = cv.QRCodeDetector()
    elif argv[1] == 'wechat':
        detector = cv.wechat_qrcode_WeChatQRCode("detect.prototxt", "detect.caffemodel", "sr.prototxt", "sr.caffemodel")
    elif argv[1] == 'analyze':
        analyze()
        return
    else:
        print("Invalid parameters! First parameter must be opencv or wechat")
        exit(-1)

    qr_detector = QRCodeDetector(mode=argv[2], detector_name=argv[1], detector=detector)
    if argv[2] == 'video':
        qr_detector.video_detect_segment()
    elif argv[2] == 'eval':
        qr_detector.images_evaluation()
    elif argv[2] == 'image' and len(argv) == 4:
        qr_detector.image_detect_segment(argv[3])
    else:
        print("Invalid parameters!")
        exit(-1)


if __name__ == "__main__":
    main(sys.argv)
