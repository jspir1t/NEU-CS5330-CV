#include <iostream>
#include <opencv2/opencv.hpp>
#include "segmentation.h"
#include "database.h"

enum mode{
  SEGMENTATION = 1,
  TRAIN = 2,
  SAVE = 3,
  CLASSIFY = 4,
} mode;
//
//static std::string get_input()
//{
//  std::string answer;
//  std::cin >> answer;
//  return answer;
//}

int main(int argc, char *argv[]) {
  // if the feature file exists, clear it at first
  clear_file();

  int component_num = 5;
  int min_area = 1000;
  mode = SEGMENTATION;
  std::fstream db_file;


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

    cv::Mat threshold_image(frame.rows, frame.cols, CV_8UC1, cv::Scalar(0));
    threshold(frame, threshold_image, THRESHOLD);

    cv::Mat cleaned_img(frame.rows, frame.cols, CV_8UC1, cv::Scalar(0));
    cleanup(threshold_image, cleaned_img, 3);

    cv::imshow("Video", frame);
    cv::imshow("Threshold", threshold_image);
    cv::imshow("cleanup", cleaned_img);

    std::map<int, cv::Mat> regions;
    cv::Mat segment_img(frame.rows, frame.cols, CV_8UC3, cv::Scalar(0));
    cv::Mat major_component(frame.rows, frame.cols, CV_8UC1, cv::Scalar(0));
    int status = segment(cleaned_img, segment_img, component_num, regions, min_area, major_component);


    // if no component detected, show the pure background image
    if (status != 0) {
      cv::imshow("components", cv::Mat(frame.rows, frame.cols, CV_8UC3, cv::Scalar(0, 0, 0)));
      continue;
    } else {
      if (mode == TRAIN) {
//        std::vector<double> feature_vector;
        std::vector<cv::Point> draw_vertices;
        single_feature_vector.clear();
        features(major_component, single_feature_vector, draw_vertices);
        mark_object(segment_img, draw_vertices);
      }
      else if (mode == SAVE) {
        std::cout << "Your are in save mode!" << std::endl;
        if (single_feature_vector.empty()) {
          std::cout << "No feature calculated" << std::endl;
          return -1;
        }
        std::cout << "Please type in the name for this object:";
        std::string feature_file_name;
        // within one thread, you cannot display windows and wait for the console input at the same time
        cv::destroyAllWindows();

        std::cin >> feature_file_name;
        write_features(db_file, feature_file_name, single_feature_vector);
        std::cout << "Save the feature successfully!" << std::endl << std::endl;
        mode = TRAIN;
        std::cout << "You are in train mode, the major component is marked" << std::endl;
        std::cout << "Press 's' to save the feature or other keys to other modes!" << std::endl;
      }
      else {
        for (std::pair<int, cv::Mat> region: regions) {
          std::vector<double> feature_vector;
          std::vector<cv::Point> draw_vertices;
          features(region.second, feature_vector, draw_vertices);
          mark_object(segment_img, draw_vertices);
          if (mode == CLASSIFY) {
            std::string label_name = euclidean_classifier(db_file, feature_vector);
            cv::putText(segment_img, label_name, draw_vertices[0], cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(200,200,250));
          }
        }
//        if (mode == CLASSIFY) {
//          euclidean_classifier()
//        }


  //      // show the top k components
  //      int window_size = (int)regions.size();
  //      cv::Mat dst = cv::Mat(frame.rows, window_size * frame.cols, CV_8UC1);
  //      int index = 0;
  //      for (auto p: regions) {
  //        p.second.copyTo(dst.rowRange(0, frame.rows).colRange(index * frame.cols, (index + 1) * frame.cols));
  //        index++;
  //      }
  //      cv::namedWindow("window", cv::WINDOW_NORMAL);
  //      cv::imshow("window", dst);
      }
      cv::imshow("components", segment_img);
    }

    // see if there is a waiting keystroke
    int key = cv::waitKey(10);
    // if user types 'q', quit the program
    if (key == 'q') {
      break;
    }
    if (key == 't') {
      mode = TRAIN;
      std::cout << "You are in train mode, the major component is marked" << std::endl;
      std::cout << "Press 's' to save the feature or space key to the segmentation mode!" << std::endl << std::endl;
    } else if (key == ' ') {
      mode = SEGMENTATION;
      std::cout << "You are in segmentation mode!" << std::endl;
    } else if (key == 's') {
      if (mode != TRAIN) {
        std::cout << "Please enter the train mode at first!" << std::endl;
      } else {
        mode = SAVE;
      }
    } else if (key == 'c') {
      if (is_empty(db_file)) {
        std::cout << "No features in database, please add some features in train mode" << std::endl;
//        std::cout << "You are in train mode, the major component is marked" << std::endl;
//        std::cout << "Press 's' to save the feature or space key to the segmentation mode!" << std::endl << std::endl;
        mode = SEGMENTATION;
      } else {
        mode = CLASSIFY;
        std::cout << "You are in classification mode!" << std::endl;
      }
    }



//    std::cout << "(" << quad_axis[0] << "," << quad_axis[2] << ") "
//              << "(" << quad_axis[1] << "," << quad_axis[2] << ") "
//              << "(" << quad_axis[0] << "," << quad_axis[3] << ") "
//              << "(" << quad_axis[1] << "," << quad_axis[3] << ") ";
//    std::cout << std::endl;



//    for (auto i: feature_vector) {
//      std::cout << i << " ";
//    }
//    std::cout << std::endl;
//    for (auto i : draw_vertices) {
//      std::cout << "(" << i.first << "," << i.second << ") ";
//    }
//    std::cout << std::endl;

  }
  delete capdev;
  return (0);
}