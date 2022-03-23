#include <opencv2/opencv.hpp>

#ifndef PROJ4_LIBRARY_H
#define PROJ4_LIBRARY_H

/**
 * Print the set along with a comment
 * @tparam T data type in the set
 * @param comment the comment string
 * @param set the set container
 */
template<typename T>
void print_set(const std::string &comment, std::vector<T> set) {
  std::cout << comment << std::endl;
  int index = 0;
  for (T point: set) {
    std::cout << point << " ";
    index++;
    if (index % 9 == 0) {
      std::cout << std::endl;
    }
  }
}

/**
 * Print the matrix along with the comment
 * @param comment the comment string
 * @param matrix the matrix represented by a cv::Mat
 */
void print_matrix(const std::string &comment, cv::Mat matrix);

/**
 * Write the camera matrix and distance coefficients into a csv file
 * @param file_name the name of csv file
 * @param camera_matrix the camera intrinsic matrix
 * @param distance_coefficients the vector of distance coefficients
 */
void write_intrinsic_paras(const std::string &file_name,
                           const cv::Mat &camera_matrix,
                           const cv::Mat &distance_coefficients);

/**
 * Read the camera matrix and distance coefficients from the csv file
 * @param file_name the name of csv file
 * @param camera_matrix the camera intrinsic matrix
 * @param distance_coefficients the vector of distance coefficients
 */
void read_intrinsic_paras(const std::string &file_name, cv::Mat &camera_matrix, cv::Mat &distance_coefficients);

/**
 * Read the corresponding .obj file based on the @param{file_path}, add the vertices and faces into the vectors
 * @param file_path the file path of the obj file
 * @param vertices the vector of vertices in obj file
 * @param faces the vector of faces in the obj file(face represented by multiple index of the vertices)
 * @param x_shift shift on the x axis, in order to make the object centered at the center of board
 * @param y_shift shift on the y axis, in order to make the object centered at the center of board
 */
void read_obj(const std::string &file_path,
              std::vector<cv::Point3f> &vertices,
              std::vector<std::vector<int>> &faces,
              float x_shift,
              float y_shift);

void draw_axes(const cv::Mat &rvec,
               const cv::Mat &tvec,
               const cv::Mat &camera_matrix,
               const cv::Mat &distance_coefficients,
               cv::Mat &frame,
               cv::Point2f origin);

void draw_object(const cv::Mat &rvec,
                 const cv::Mat &tvec,
                 const cv::Mat &camera_matrix,
                 const cv::Mat &distance_coefficients,
                 const std::vector<cv::Point3f> &vertices,
                 std::vector<std::vector<int>> &faces,
                 cv::Mat &frame
);

#endif //PROJ4_LIBRARY_H
