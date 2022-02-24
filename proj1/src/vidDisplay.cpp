#include <iostream>
#include <opencv2/opencv.hpp>
#include "filter.h"

using namespace cv;

enum effect {
  ORIGIN = 1,
  GRAY = 2,
  ALTERNATIVE_GREY = 3,
  BLUR = 4,
  X_SOBEL = 5,
  Y_SOBEL = 6,
  MAGNITUDE = 7,
  BLUR_QUANTIZE = 8,
  CARTOON = 9,
  NEGATIVE = 10,
  ADD_CONTRAST = 11,
  DEC_CONTRAST = 12,
  ADD_BRIGHTNESS = 13,
  DEC_BRIGHTNESS = 14,
  SEPIA = 15,
  EMBOSS = 16,
} effect;

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
  cv::namedWindow("Video", 1); // identifies a window
  cv::Mat frame;
  cv::Mat converted_frame(refS.height, refS.width, CV_8UC3);
  int quantize_level = 15;
  int threshold = 20;
  effect = ORIGIN;
  for (;;) {
    *capdev >> frame; // get a new frame from the camera, treat as a stream
    if (frame.empty()) {
      printf("frame is empty\n");
      break;
    }
    // see if there is a waiting keystroke
    int key = cv::waitKey(10);
    // if user types 'q', quit the program
    if (key == 'q') {
      break;
    }
    // if user types 's', save the open image to img/screenshot.png
    if (key == 's') {
      cv::imwrite("../img/screenshot.png", converted_frame);
    }
    switch (key) {
      case 'g':effect = GRAY;
        break;
      case 'b':effect = BLUR;
        break;
      case 'h':effect = ALTERNATIVE_GREY;
        break;
      case 'x':effect = X_SOBEL;
        break;
      case 'y':effect = Y_SOBEL;
        break;
      case 'm':effect = MAGNITUDE;
        break;
      case 'l':effect = BLUR_QUANTIZE;
        break;
      case 'c':effect = CARTOON;
        break;
      case 'n':effect = NEGATIVE;
        break;
      case '<':effect = DEC_CONTRAST;
        break;
      case '>':effect = ADD_CONTRAST;
        break;
      case '+':effect = ADD_BRIGHTNESS;
        break;
      case '-':effect = DEC_BRIGHTNESS;
        break;
      case 'p':effect = SEPIA;
        break;
      case 'e':effect = EMBOSS;
        break;
      case ' ':effect = ORIGIN;
        break;
    }

    switch (effect) {
      case GRAY: {
        cv::Mat converted_frame_one_channel(refS.height, refS.width, CV_8UC1);
        // cvtColor will make the dst a 1 channel mat, which is fine for display but will cause the program crash
        // when we switch to other mode immediately. Thus, I duplicate the channel to 3 channels mat with cv::merge()
        cv::cvtColor(frame.clone(), converted_frame_one_channel, COLOR_BGR2GRAY);
        std::vector<cv::Mat> channels;
        channels.push_back(converted_frame_one_channel);
        channels.push_back(converted_frame_one_channel);
        channels.push_back(converted_frame_one_channel);
        cv::merge(channels, converted_frame);
        break;
      }
      case ALTERNATIVE_GREY: {
        greyscale(frame, converted_frame);
        break;
      }
      case BLUR: {
        blur5x5(frame, converted_frame);
        break;
      }
      case X_SOBEL: {
        cv::Mat frame_16s(frame.rows, frame.cols, CV_16SC3);
        sobelX3x3(frame, frame_16s);
        transform(frame_16s, converted_frame);
        break;
      }
      case Y_SOBEL: {
        cv::Mat frame_16s(frame.rows, frame.cols, CV_16SC3);
        sobelY3x3(frame, frame_16s);
        transform(frame_16s, converted_frame);
        break;
      }
      case MAGNITUDE: {
        cv::Mat x_sobel_frame_16s(frame.rows, frame.cols, CV_16SC3);
        sobelX3x3(frame, x_sobel_frame_16s);
        cv::Mat y_sobel_frame_16s(frame.rows, frame.cols, CV_16SC3);
        sobelY3x3(frame, y_sobel_frame_16s);
        magnitude(x_sobel_frame_16s, y_sobel_frame_16s, converted_frame);
        break;
      }
      case BLUR_QUANTIZE: {
        blurQuantize(frame, converted_frame, quantize_level);
        break;
      }
      case CARTOON: {
        cartoon(frame, converted_frame, quantize_level, threshold);
        break;
      }
      case NEGATIVE: {
        negative(frame, converted_frame);
        break;
      }
      case ADD_CONTRAST: {
        frame.convertTo(converted_frame, -1, 2, 0);
        break;
      }
      case DEC_CONTRAST: {
        frame.convertTo(converted_frame, -1, 0.25, 0);
        break;
      }
      case ADD_BRIGHTNESS: {
        frame.convertTo(converted_frame, -1, 1, 100);
        break;
      }
      case DEC_BRIGHTNESS: {
        frame.convertTo(converted_frame, -1, 1, -100);
        break;
      }
      case SEPIA: {
        sepia(frame, converted_frame);
        break;
      }
      case EMBOSS: {
        emboss(frame, converted_frame);
        break;
      }
      default: {
        frame.copyTo(converted_frame);
        break;
      }
    }
    cv::imshow("Video", converted_frame);
  }
  delete capdev;
  return (0);
}