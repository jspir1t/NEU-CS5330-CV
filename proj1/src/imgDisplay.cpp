#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char *argv[]) {
  Mat image = imread("../img/test.png");
  // Create a window for display.
  cv::namedWindow("Window", WINDOW_AUTOSIZE);
  cv::imshow("Window", image);
  while (true) {
    // check for a key press
    int key = waitKey(0);
    // if user types 'q', quit the program
    if (key == 'q') {
      break;
    }
    // if user types 's', save the open image to img/copy.png
    if (key == 's') {
      cv::imwrite("../img/copy.png", image);
    }
  }
  return 0;
}
