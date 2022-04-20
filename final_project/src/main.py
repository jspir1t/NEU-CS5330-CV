import cv2 as cv
import os
import sys

os.chdir("../config")


def draw_bounding_box(mat, box):
    for step in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        start = box[step[0]][0], box[step[0]][1]
        end = box[step[1]][0], box[step[1]][1]
        cv.line(mat, start, end, (0, 255, 0))


class QRCodeDetector:
    def __init__(self, mode, detector_name, detector):
        self.mode = mode
        self.detector_name = detector_name
        self.detector = detector

    def image_detect_segment(self, img_name):
        img_dir = f'../images/{img_name}.png'
        if not os.path.exists(img_dir):
            print("No such image in images folder!")
            exit(-1)

        img = cv.imread(img_dir)
        contents, points, *malicious = self.detector.detectAndDecode(img)
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
        src_folder = '../images'
        dst_folder = '../results'
        results = {}
        for img_name in os.listdir(src_folder):
            img_path = os.path.join(src_folder, img_name)
            img = cv.imread(img_path)
            contents, points, *malicious = self.detector.detectAndDecode(img)
            if len(contents) > 0:
                results[img_name] = True
                for box in points:
                    draw_bounding_box(img, box.astype(int))
            else:
                results[img_name] = False
            cv.imwrite(f'{dst_folder}/{self.mode}_{self.detector_name}_{img_name}', img)
        print(results)

    def video_detect_segment(self):
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
    print(
        'Usage: python main.py opencv/wechat video | python main.py opencv/wechat image <image_name> | python main.py '
        'opencv/wechat eval')
    detector = None
    qr_detector = None

    if argv is None or len(argv) > 4:
        print("Invalid parameters!")
        exit(-1)

    if argv[1] == 'opencv':
        detector = cv.QRCodeDetector()
    elif argv[1] == 'wechat':
        detector = cv.wechat_qrcode_WeChatQRCode("detect.prototxt", "detect.caffemodel", "sr.prototxt", "sr.caffemodel")
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
