#include <opencv2/opencv.hpp>

#ifndef PROJ3_INCLUDE_SEGEMENTATION_H_
#define PROJ3_INCLUDE_SEGEMENTATION_H_

#define BACK_GROUND 0
#define FRONT_GROUND 255
#define THRESHOLD 100

int threshold(const cv::Mat &src, cv::Mat &dst, int threshold);
int cleanup(cv::Mat &src, cv::Mat &dst, int steps);
int segment(cv::Mat &src, cv::Mat &dst, int component_num, std::map<int, cv::Mat> &regions, int min_area, cv::Mat &major);
void mark_object(cv::Mat &segmented_img, std::vector<cv::Point> draw_vertices);
int features(cv::Mat &src, std::vector<double> &feature_vector, std::vector<cv::Point> &draw_vertices);

#endif //PROJ3_INCLUDE_SEGEMENTATION_H_
