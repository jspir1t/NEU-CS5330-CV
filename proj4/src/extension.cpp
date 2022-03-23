#include <iostream>
#include "utils.h"


int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cout << "Usage: ./extension.exe <obj_name>" << std::endl;
    exit(-1);
  }

  cv::SimpleBlobDetector::Params params;
  params.maxArea = 10e4;
  params.minArea = 10;
  params.minDistBetweenBlobs = 5;
  // Initialize the simple blob detector with the parameters above for the findCirclesGrid() function to find the circle grids
  cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

  std::vector<cv::Point2f> corner_set;
  std::vector<cv::Vec3f> point_set;
  std::vector<std::vector<cv::Vec3f> > point_list;
  std::vector<std::vector<cv::Point2f> > corner_list;

  cv::TermCriteria termcrit(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,20,0.03);
  cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64FC1);
  cv::Mat distortion_coefficients;
  // The circles feature we use is 4 * 11
  int points_per_row = 4 ,points_per_colum = 11;
  cv::Size pattern_size = cv::Size(points_per_row, points_per_colum);
  int count = 0;
  bool show_axes = false;
  std::vector<cv::Point3f> vertices;
  std::vector<std::vector<int>> faces;
  read_obj("../objs/" + std::string(argv[1]) + ".obj", vertices, faces, 3.5f, -5.f);

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
  std::string delimiter(50, '-');

  for (;;) {
    *capdev >> frame; // get a new frame from the camera, treat as a stream
    if (frame.empty()) {
      printf("frame is empty\n");
      break;
    }

    cv::cvtColor(frame, grey_scale, cv::COLOR_BGR2GRAY);

    corner_set.clear();
    point_set.clear();

    bool pattern_found = cv::findCirclesGrid(frame, pattern_size, corner_set, cv::CALIB_CB_ASYMMETRIC_GRID, detector);
    if (pattern_found) {
      cv::cornerSubPix(grey_scale, corner_set, cv::Size(5, 5), cv::Size(-1, -1), termcrit);
      cv::drawChessboardCorners(frame, pattern_size, corner_set, pattern_found);
      // draw the origin point in the circle grid
      cv::circle(frame, corner_set[0], 20, cv::Scalar(255, 255, 0));
    }

    // see if there is a waiting keystroke
    int key = cv::waitKey(10);
    // if user types 'q', quit the program
    if (key == 'q') {
      break;
    }
    if (key == 's') {
      if (!pattern_found) {
        std::cout << "No circle found!" << std::endl;
      } else {
        std::cout << delimiter << std::endl;
        std::cout << "calibration frame " + std::to_string(count) + " saved." << std::endl;

        // generate the point set, examples:
        // (0, 0, 0) (2, 0, 0) (4, 0, 0) (6, 0, 0)
        // (1, -1, 0) (3, -1, 0) (5, -1, 0) (7, -1, 0)
        for (int i = 0; i < points_per_colum; i++) {
          int j = (i % 2) == 0 ? 0 : 1;
          for (; j < points_per_row * 2; j = j + 2) {
            point_set.push_back(cv::Point3f((float)j, (float)(-i), 0.f));
          }
        }
        corner_list.push_back(std::vector<cv::Point2f>(corner_set));
        point_list.push_back(std::vector<cv::Vec3f>(point_set));

        print_set("adding corner set" + std::to_string(count) + " to the corner list:", point_set);
        print_set("", corner_set);

        cv::imwrite("../img/circles/screenshot_" + std::to_string(count++) + ".jpg", frame);
      }
    } else if (key == 'c') {
      if (point_list.size() >= 5) {
        camera_matrix.at<double>(0, 0) = 1;
        camera_matrix.at<double>(1, 1) = 1;
        camera_matrix.at<double>(0, 2) = frame.cols / 2;
        camera_matrix.at<double>(1, 2) = frame.cols / 2;
        std::cout << delimiter << std::endl;
        std::cout << "5 frame collected, running a calibration:" << std::endl;
        print_matrix("initial camera matrix: ", camera_matrix);

        std::vector<cv::Mat> rvecs, tvecs;
        double re_projection_error = cv::calibrateCamera(point_list, corner_list, frame.size(), camera_matrix, distortion_coefficients, rvecs, tvecs, cv::CALIB_FIX_ASPECT_RATIO, termcrit);

        print_matrix("camera_matrix: ", camera_matrix);
        print_matrix("distortion_coefficients:", distortion_coefficients);
        std::cout << "re-projection error: " << re_projection_error << std::endl;
        std::cout << "save camera matrix and distortion coefficients to csv file." << std::endl;
        std::cout << delimiter << std::endl;
      } else {
        std::cout << "calibration frames not enough: require 5, current " + std::to_string(point_list.size()) << std::endl;
      }
    } else if (key == 'w' || show_axes) {
      // In extension, instead of reading the intrinsic parameters from the file storage, I simply use the results from calibration above.
      if (point_list.size() >= 5 && corner_set.size() == 44) {
        show_axes = true;
        for (int i = 0; i < points_per_colum; i++) {
          int j = (i % 2) == 0 ? 0 : 1;
          for (; j < points_per_row * 2; j = j + 2) {
            point_set.push_back(cv::Point3f((float)j, (float)(-i), 0.f));
          }
        }

        cv::Mat rvec, tvec;
        // The file storage is not implemented in extension, it simply uses the camera_matrix and distortion coefficients above
        cv::solvePnP(point_set, corner_set, camera_matrix, distortion_coefficients, rvec, tvec);
//        std::cout << "rotation matrix: " << std::endl << rvec << std::endl;
//        std::cout << "translation matrix: " << std::endl << tvec << std::endl;

        // draw axes
        draw_axes(rvec, tvec, camera_matrix, distortion_coefficients, frame, corner_set[0]);

        // show 3d objects
        draw_object(rvec, tvec, camera_matrix, distortion_coefficients, vertices, faces, frame);
      } else {
        if (point_list.size() < 5) {
          std::cout << "calibration frames not enough: require 5, current " + std::to_string(point_list.size()) << std::endl;
        } else {
          std::cout << "Circles are not detected correctly!" << std::endl;
        }
      }
    }
    cv::imshow("Video", frame);
  }
  delete capdev;
  return (0);
}