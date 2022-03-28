#include <iostream>
#include "utils.h"

enum mode{
  HARRIS = 1,
  SIFT = 2,
};

int main(int argc, char *argv[]) {
  mode m = HARRIS;

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
  cv::Mat grey_scale;
  cv::Mat dst = cv::Mat::zeros(refS.width, refS.height, CV_32FC1);
  cv::Mat dst_norm, dst_norm_scaled;

  for (;;) {
    *capdev >> frame; // get a new frame from the camera, treat as a stream
    if (frame.empty()) {
      printf("frame is empty\n");
      break;
    }

    cv::cvtColor(frame, grey_scale, cv::COLOR_BGR2GRAY);

    if (m == HARRIS) {
      int blockSize = 5;
      int apertureSize = 3;
      double k = 0.04;
      cornerHarris(grey_scale, dst, blockSize, apertureSize, k);
      normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
      convertScaleAbs(dst_norm, dst_norm_scaled);
      for (int i = 0; i < dst_norm.rows; i++) {
        for (int j = 0; j < dst_norm.cols; j++) {
          if ((int) dst_norm.at<float>(i, j) > 100) {
            circle(frame, cv::Point(j, i), 1, cv::Scalar(0, 0, 255), 1, 8, 0);
          }
        }
      }
    } else {
      cv::Ptr<cv::SiftFeatureDetector> detector = cv::SiftFeatureDetector::create();
      std::vector<cv::KeyPoint> key_points;
      detector->detect(grey_scale, key_points);

      cv::Mat output;
      cv::drawKeypoints(grey_scale, key_points, output, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
      output.copyTo(frame);
    }



    // see if there is a waiting keystroke
    int key = cv::waitKey(10);
    // if user types 'q', quit the program
    if (key == 'q') {
      break;
    }
    if (key == 's') {
      cv::imwrite("../corners.jpg", frame);
    }
    // if user types 'h', show the harris corners
    else if (key == 'h') {
      m = HARRIS;
    }
      // if user types 'f', show the sift corners with orientations
    else if (key == 'f') {
      m = SIFT;
    }

    imshow("Video", frame);
  }
  delete capdev;
  return (0);
}