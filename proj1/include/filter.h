#include <opencv2/opencv.hpp>

#ifndef PROJ1__FILTERS_H_
#define PROJ1__FILTERS_H_
int greyscale(cv::Mat &src, cv::Mat &dst);

int blur5x5(cv::Mat &src, cv::Mat &dst);

int sobelX3x3(cv::Mat &src, cv::Mat &dst);

int sobelY3x3(cv::Mat &src, cv::Mat &dst);

int transform(cv::Mat &src, cv::Mat &dst);

int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);

int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);

int cartoon(cv::Mat &src, cv::Mat &dst, int levels, int magThreshold);

int negative(cv::Mat &src, cv::Mat &dst);

int sepia(cv::Mat &src, cv::Mat &dst);

int emboss(cv::Mat &src, cv::Mat &dst);
#endif //PROJ1__FILTERS_H_
