#include "utils.h"

using namespace cv;
using namespace std;

enum mode {
  EVAL = 1,
  VIDEO = 2,
  IMAGE = 3
} mode;

int videoDetectSegment() {
  cv::VideoCapture *capdev;
  // open the video device
  capdev = new cv::VideoCapture(0);
  if (!capdev->isOpened()) {
    printf("Unable to open video device\n");
    return (-1);
  }
  // get some properties of the image
  cv::Size refS((int) capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
  printf("Expected size: %d %d\n", refS.width, refS.height);
  cv::namedWindow("Video", 1); // identifies a window
  cv::Mat frame;
  std::vector<double> single_feature_vector;
  for (;;) {
    *capdev >> frame; // get a new frame from the camera, treat as a stream
    if (frame.empty()) {
      printf("frame is empty\n");
      break;
    }
    detectAndSegmentQRCode(frame);
    // see if there is a waiting keystroke
    int key = cv::waitKey(10);
    // if user types 'q', quit the program
    if (key == 'q') {
      break;
    }
    if (key == 's') {
      String file_name = "../results/" + to_string(mode) + ".png";
      imwrite(file_name, frame);
      cout << "This image is saved as " << file_name << endl;
    }
  }
}

int main(int argc, char *argv[]) {
  mode = VIDEO;
  if (argc < 2 || argc >= 4) {
    std::cout << "Usage: ./main.exe video/eval || ./main.exe image <image_name>" << std::endl;
    exit(-1);
  }
  String m = argv[1];
  if (m == "video") {
    videoDetectSegment();
    mode = VIDEO;
  } else if (m == "eval") {
    mode = EVAL;
  } else if (m == "image" && argc == 3) {
    mode = IMAGE;
  } else {
    cout << "Invalid parameters!" << endl;
    exit(-1);
  }

//  Mat src = imread("../images/qrcode.png");
//  if (src.empty()) {
//    printf("could not load image file...");
//    return -1;
//  }
//  imshow("input", src);
//  detectAndSegmentQRCode(src);
//  waitKey(0);
  return 0;
}