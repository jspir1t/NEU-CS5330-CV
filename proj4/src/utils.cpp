#include <fstream>
#include "utils.h"

void print_matrix(const std::string &comment, cv::Mat matrix) {
  std::cout << comment << std::endl;
  for (int i = 0; i < matrix.rows; i++) {
    for (int j = 0; j < matrix.cols; j++) {
      std::cout << matrix.at<double>(i, j) << "\t";
    }
    std::cout << std::endl;
  }
}

void write_intrinsic_paras(const std::string &file_name,
                           const cv::Mat &camera_matrix,
                           const cv::Mat &distortion_coefficients) {
  std::ofstream intrinsic_file;
  intrinsic_file.open("../" + file_name);
  intrinsic_file.flush();

  std::string header = "cm_00,cm_01,cm_02,cm_10,cm_11,cm_12,cm_20,cm_21,cm_22,";
  for (int i = 0; i < distortion_coefficients.rows * distortion_coefficients.cols; i++) {
    header += "dc_" + std::to_string(i) + ",";
  }
  header += "\n";
  intrinsic_file << header;

  for (int i = 0; i < camera_matrix.rows; i++) {
    for (int j = 0; j < camera_matrix.cols; j++) {
      intrinsic_file << camera_matrix.at<double>(i, j) << ",";
    }
  }

  for (int i = 0; i < distortion_coefficients.rows; i++) {
    for (int j = 0; j < distortion_coefficients.cols; j++) {
      intrinsic_file << distortion_coefficients.at<double>(i, j) << ",";
    }
  }
  intrinsic_file.close();
}

void read_intrinsic_paras(const std::string &file_name, cv::Mat &camera_matrix, cv::Mat &distortion_coefficients) {
  std::fstream intrinsic_file;
  intrinsic_file.open("../" + file_name);

  std::vector<double> result;
  std::string line;
  if (!std::getline(intrinsic_file, line, '\n')) {
    std::cout << "No content in the csv file" << std::endl;
    exit(-1);
  }
  std::getline(intrinsic_file, line, '\n');

  camera_matrix = cv::Mat::zeros(3, 3, CV_64FC1);
  distortion_coefficients = cv::Mat(1, 5, CV_64FC1);
  std::stringstream ss(line);
  while (ss.good()) {
    std::string substr;
    std::getline(ss, substr, ',');
    if (!substr.empty())
      result.push_back(std::strtod(substr.c_str(), nullptr));
  }

  int index = 0;
  for (int i = 0; i < camera_matrix.rows; i++) {
    for (int j = 0; j < camera_matrix.cols; j++) {
      camera_matrix.at<double>(i, j) = result.at(index++);
    }
  }
  for (int i = 0; i < distortion_coefficients.rows; i++) {
    for (int j = 0; j < distortion_coefficients.cols; j++) {
      distortion_coefficients.at<double>(i, j) = result.at(index++);
    }
  }
  intrinsic_file.close();
}

void read_obj(const std::string &file_path,
              std::vector<cv::Point3f> &vertices,
              std::vector<std::vector<int>> &faces,
              float x_shift,
              float y_shift) {
  std::string line;
  std::vector<std::string> values;
  std::ifstream file(file_path);
  if (!file.good()) {
    std::cout << "No such file" << std::endl;
    std::cout << "Candidate objects: [humanoid, teapot, teddy], Please refer to the objs directory";
    exit(-1);
  }
  while (std::getline(file, line)) {
    values.clear();

    // split the line
    std::stringstream ss(line);
    while (ss.good()) {
      std::string substr;
      std::getline(ss, substr, ' ');
      if (!substr.empty())
        values.push_back(substr);
    }
    if (values.empty()) {
      continue;
    }
    if (values.size() != 4) {
      std::cout << "FILE FORMAT ERROR" << std::endl;
      exit(-1);
    }

    if (values.at(0) == "v") {
      // add x_shift to x and y_shift to y, which could shift the 3d object to the middle of the chessboard
      vertices.push_back(cv::Point3f(std::stof(values.at(1)) + x_shift,
                                     std::stof(values.at(2)) + y_shift,
                                     std::stof(values.at(3))));
    } else if (values.at(0) == "f") {
      faces.push_back({std::stoi(values.at(1)), std::stoi(values.at(2)), std::stoi(values.at(3))});
    } else {
      std::cout << "FILED NOT SUPPORT NOW!" << std::endl;
      exit(-1);
    }
  }
  std::cout << "vertices number: " + std::to_string(vertices.size())
            << ", faces number: " + std::to_string(faces.size()) << std::endl;
}

void draw_axes(const cv::Mat &rvec,
               const cv::Mat &tvec,
               const cv::Mat &camera_matrix,
               const cv::Mat &distortion_coefficients,
               cv::Mat &frame,
               cv::Point2f origin) {
  std::vector<cv::Point3f> object_points;
  object_points.emplace_back(3., 0., 0.);
  object_points.emplace_back(0., -3., 0.);
  object_points.emplace_back(0., 0., 3.);
  std::vector<cv::Point2f> image_points;
  cv::projectPoints(object_points, rvec, tvec, camera_matrix, distortion_coefficients, image_points);
  cv::arrowedLine(frame, origin, image_points[0], cv::Scalar(255, 0, 0), 2);
  cv::arrowedLine(frame, origin, image_points[1], cv::Scalar(0, 255, 0), 2);
  cv::arrowedLine(frame, origin, image_points[2], cv::Scalar(0, 0, 255), 2);
}

void draw_object(const cv::Mat &rvec,
                 const cv::Mat &tvec,
                 const cv::Mat &camera_matrix,
                 const cv::Mat &distortion_coefficients,
                 const std::vector<cv::Point3f> &vertices,
                 std::vector<std::vector<int>> &faces,
                 cv::Mat &frame
) {
  // show 3d objects
  std::vector<cv::Point2f> object_image_points;
  cv::projectPoints(vertices, rvec, tvec, camera_matrix, distortion_coefficients, object_image_points);
  for (std::vector<int> &face: faces) {
    cv::line(frame,
             object_image_points[face[0] - 1],
             object_image_points[face[1] - 1],
             cv::Scalar(0, 255, 0),
             1);
    cv::line(frame,
             object_image_points[face[1] - 1],
             object_image_points[face[2] - 1],
             cv::Scalar(0, 255, 0),
             1);
    // draw the line between the last vertices and the first vertices in a face
    cv::line(frame,
             object_image_points[face[2] - 1],
             object_image_points[face[0] - 1],
             cv::Scalar(0, 255, 0),
             1);
  }
}