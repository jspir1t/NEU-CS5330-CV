#include <iostream>
#include "utils.h"

cv::Mat frame;
cv::Mat grey_scale;
int corner_num = 100;

void cornerHarris( int, void* );

int main(int argc, char *argv[]) {
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
  cv::namedWindow("Source", 1); // identifies a window
  cv::namedWindow( "Corners" );
  cv::createTrackbar( "Corner Number: ", "Source", &corner_num, 300, cornerHarris );
  cv::Mat dst = cv::Mat::zeros(refS.width, refS.height, CV_32FC1);
  cv::Mat dst_norm, dst_norm_scaled;

  for (;;) {
    *capdev >> frame; // get a new frame from the camera, treat as a stream
    if (frame.empty()) {
      printf("frame is empty\n");
      break;
    }

    cv::cvtColor(frame, grey_scale, cv::COLOR_BGR2GRAY);
    imshow( "Source", frame );

    cornerHarris( 0, nullptr );

    // see if there is a waiting keystroke
    int key = cv::waitKey(10);
    // if user types 'q', quit the program
    if (key == 'q') {
      break;
    }
  }
  delete capdev;
  return (0);
}

void cornerHarris( int, void* )
{
  if (frame.empty()) {
    return;
  }
//  cv::Mat frame_copy = frame.clone();
  std::vector<cv::Point2f> corners;
  cv::goodFeaturesToTrack(grey_scale, corners, corner_num, 0.05, 1, cv::noArray(), 3, true);
  for(cv::Point2f point: corners) {
    circle( frame, point, 3,  cv::Scalar(0, 255, 0), 2, 8, 0 );
  }
  cv::putText(frame, "Corner Number: " + std::to_string(corners.size()), cv::Point(0, 20), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,0,0), 2);


//  int blockSize = 5;
//  int apertureSize = 3;
//  double k = 0.04;
//  cv::Mat dst = cv::Mat::zeros( frame.size(), CV_32FC1 );
//  cornerHarris( grey_scale, dst, blockSize, apertureSize, k );
//  cv::Mat dst_norm, dst_norm_scaled;
//  normalize( dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );
//  convertScaleAbs( dst_norm, dst_norm_scaled );
//  for( int i = 0; i < dst_norm.rows ; i++ )
//  {
//    for( int j = 0; j < dst_norm.cols; j++ )
//    {
//      if( (int) dst_norm.at<float>(i,j) > threshold )
//      {
//        circle( frame_copy, cv::Point(j,i), 2,  cv::Scalar(0, 255, 0), 1, 8, 0 );
//      }
//    }
//  }
  cv::namedWindow( "Corners" );
  imshow( "Corners", frame );
}