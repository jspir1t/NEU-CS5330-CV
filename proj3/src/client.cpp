#include <iostream>
#include <opencv2/opencv.hpp>
#include "retrieval.h"
#include "classifier.h"

enum mode {
  SEGMENTATION = 1,
  TRAIN_DATA_PREP = 2,
  TRAIN_SAVE = 3,
  NN = 4,
  KNN = 5,
  TEST_DATA_PREP = 6,
  EVALUATE_SAVE = 7,
  EVALUATE = 8,
} mode;

int main(int argc, char *argv[]) {
  // if the feature file exists and evaluation result file exist, clear them at first
  clear_file(FEATURE_FILE_NAME);
  clear_file(EVALUATE_FILE_NAME);

  // number of component shown, ordered by area size
  int component_num = 5;
  // minimum size of area to be shown
  int min_area = 1000;
  mode = SEGMENTATION;
  std::fstream db_file;
  std::fstream test_file;
  // set up k for knn
  int k = 3;
  // set up steps to shrink or grow
  int steps = 5;

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

    // thresholding the image
    cv::Mat threshold_image(frame.rows, frame.cols, CV_8UC1, cv::Scalar(0));
    threshold(frame, threshold_image, THRESHOLD);

    // clean the image with shrink and grow
    cv::Mat cleaned_img(frame.rows, frame.cols, CV_8UC1, cv::Scalar(0));
    cleanup(threshold_image, cleaned_img, steps);

    cv::imshow("Video", frame);
    cv::imshow("Threshold", threshold_image);
    cv::imshow("cleanup", cleaned_img);

    // segment the mat into different components, the major component is the one with the largest area(excluding background)
    std::map<int, cv::Mat> regions;
    cv::Mat components_img(frame.rows, frame.cols, CV_8UC3, cv::Scalar(0));
    cv::Mat major_component(frame.rows, frame.cols, CV_8UC1, cv::Scalar(0));
    int status = segment(cleaned_img, components_img, component_num, regions, min_area, major_component);

    int window_size = (int)regions.size();
    cv::Mat segmentation_img = cv::Mat(frame.rows, window_size * frame.cols, CV_8UC1);

    // if no component detected, show the pure background image with a sign "No component detected"
    if (status != 0) {
      cv::Mat pure_black = cv::Mat(frame.rows, frame.cols, CV_8UC3, cv::Scalar(0));
      cv::putText(pure_black,
                  "No component detected",
                  cv::Point(frame.rows / 2, frame.cols / 2),
                  cv::FONT_HERSHEY_COMPLEX_SMALL,
                  1.0,
                  cv::Scalar(0, 0, 255));
      cv::imshow("components", pure_black);
      cv::imshow("segmentation", pure_black);
    } else {
      // if you are in training data or test data preparation, it will only mark the major component
      if (mode == TRAIN_DATA_PREP || mode == TEST_DATA_PREP) {
        std::vector<cv::Point> draw_vertices;
        single_feature_vector.clear();
        features(major_component, single_feature_vector, draw_vertices);
        mark_object(components_img, draw_vertices);
      } else if (mode == TRAIN_SAVE) {
        std::cout << "Saving current feature for training..." << std::endl;
        std::cout << "Please type in the name for this object:";
        std::string feature_name;
        // within one thread, you cannot display windows and wait for the console input at the same time
        cv::destroyAllWindows();

        std::cin >> feature_name;
        write_features(db_file, feature_name, single_feature_vector, FEATURE_FILE_NAME);
        std::cout << "Save the feature successfully!" << std::endl << std::endl;
        mode = TRAIN_DATA_PREP;
        std::cout << "Back to TRAIN_DATA_PREP mode, the major component is marked" << std::endl;
        std::cout << "Press 's' to save the feature or other keys to other modes!" << std::endl;
      } else if (mode == EVALUATE_SAVE) {
        std::cout << "Saving current feature for evaluating..." << std::endl;
        std::cout << "Please type in the name for this object:";
        std::string feature_name;
        // within one thread, you cannot display windows and wait for the console input at the same time
        cv::destroyAllWindows();

        std::cin >> feature_name;
        write_features(test_file, feature_name, single_feature_vector, EVALUATE_FILE_NAME);
        std::cout << "Save the feature successfully!" << std::endl << std::endl;
        mode = TEST_DATA_PREP;
        std::cout << "Back to TEST_DATA_PREP mode, the major component is marked" << std::endl;
        std::cout << "Press 's' to save the feature or other keys to other modes!" << std::endl;
      } else {
        // if you are in the default segmentation mode, it will show all the components in the window
        for (std::pair<int, cv::Mat> region: regions) {
          std::vector<double> feature_vector;
          std::vector<cv::Point> draw_vertices;
          features(region.second, feature_vector, draw_vertices);
          mark_object(components_img, draw_vertices);
          // show the result from NN algorithm in the mat
          if (mode == NN) {
            std::string label_name = nearest_neighbor_classifier(db_file, feature_vector);
            cv::putText(components_img,
                        "NN: " + label_name,
                        draw_vertices[0],
                        cv::FONT_HERSHEY_COMPLEX_SMALL,
                        0.8,
                        cv::Scalar(200, 200, 250));
          }
            // show the result from KNN algorithm in the mat
          else if (mode == KNN) {
            std::string label_name;
            int status = knn_classifier(db_file, feature_vector, k, label_name);
            if (status == -1) {
              std::cout << "At least " << k << " examples in each class!" << std::endl;
              std::cout << "Back to SEGMENTATION mode!" << std::endl;
              mode = SEGMENTATION;
            }
            cv::putText(components_img,
                        "KNN: " + label_name,
                        draw_vertices[0],
                        cv::FONT_HERSHEY_COMPLEX_SMALL,
                        0.8,
                        cv::Scalar(200, 200, 250));
          }
            // evaluate the features in "../test_features.txt" and write the confusion matrix in "../evaluation.txt"
          else if (mode == EVALUATE) {
            evaluate(db_file, test_file, k);
            mode = SEGMENTATION;
          }
        }
      }
      // show the top k components
      int index = 0;
      for (auto p: regions) {
        p.second.copyTo(segmentation_img.rowRange(0, frame.rows).colRange(index * frame.cols, (index + 1) * frame.cols));
        index++;
      }
//        cv::imshow("segmentation", segmentation_img);
      cv::imshow("segmentation", segmentation_img);
      cv::imshow("components", components_img);
    }

    // see if there is a waiting keystroke
    int key = cv::waitKey(10);
    // if user types 'q', quit the program
    if (key == 'q') {
      break;
    }
    if (key == 't') {
      mode = TRAIN_DATA_PREP;
      std::cout << "You are in TRAIN_DATA_PREP mode, the major component is marked" << std::endl;
      std::cout << "Press 's' to save the feature or space key to the SEGMENTATION mode!" << std::endl << std::endl;
    } else if (key == ' ') {
      mode = SEGMENTATION;
      std::cout << "You are in segmentation mode!" << std::endl;
    } else if (key == 's') {
      if (mode != TRAIN_DATA_PREP && mode != TEST_DATA_PREP) {
        std::cout << "Please enter the TRAIN_DATA_PREP mode or TEST_DATA_PREP mode at first!" << std::endl;
      } else if (mode == TRAIN_DATA_PREP) {
        mode = TRAIN_SAVE;
      } else {
        mode = EVALUATE_SAVE;
      }
    } else if (key == 'n') {
      if (is_empty(db_file, FEATURE_FILE_NAME)) {
        std::cout << "No features in database, please add some features in TRAIN_DATA_PREP mode" << std::endl;
        mode = SEGMENTATION;
      } else {
        mode = NN;
        std::cout << "You are in NEAREST_NEIGHBOR_CLASSIFY mode!" << std::endl;
      }
    } else if (key == 'k') {
      if (is_empty(db_file, FEATURE_FILE_NAME)) {
        std::cout << "No features in database, please add some features in TRAIN_DATA_PREP mode" << std::endl;
        mode = SEGMENTATION;
      } else {
        mode = KNN;
        std::cout << "You are in KNN mode!" << std::endl;
      }
    } else if (key == 'p') {
      mode = TEST_DATA_PREP;
      std::cout << "You are in TEST_DATA_PREP mode!" << std::endl;
      std::cout << "Press 's' to save the feature or space key to the SEGMENTATION mode!" << std::endl << std::endl;
    } else if (key == 'e') {
      if (is_empty(db_file, EVALUATE_FILE_NAME)) {
        std::cout << "No features to be evaluated, please add some features in TEST_DATA_PREP mode" << std::endl;
        mode = SEGMENTATION;
      } else {
        mode = EVALUATE;
        std::cout << "You are in EVALUATE mode!" << std::endl;
        std::cout << "Confusion Matrix generated in: " << EVALUATE_OUTPUT_FILE_NAME << std::endl;
      }
    } else if (key == 'a') {
      cv::imwrite("../origin.jpg", frame);
      cv::imwrite("../threshold.jpg", threshold_image);
      cv::imwrite("../cleanup.jpg", cleaned_img);
      cv::imwrite("../segmentation.jpg", segmentation_img);
      cv::imwrite("../components.jpg", components_img);
    }
  }
  delete capdev;
  return (0);
}