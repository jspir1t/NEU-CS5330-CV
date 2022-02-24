//
// Created by stard on 2022/2/7.
//
#include "opencv2/opencv.hpp"

#ifndef PROJ2_INCLUDE_FEATURES_H_
#define PROJ2_INCLUDE_FEATURES_H_

#define BINS 16
#define BASELINE_IMAGE "..\\olympus\\pic.1016.jpg"
#define COLOR_HIST_IMAGE "..\\olympus\\pic.0164.jpg"
#define MULTI_HIST_IMAGE "..\\olympus\\pic.0274.jpg"
#define TEXTURE_COLOR_HIST_IMAGE "..\\olympus\\pic.0535.jpg"
// if you want to see the effect of blue bins, change it to #define CUSTOM_HIST_IMAGE "..\\olympus\\pic.0287.jpg"
#define CUSTOM_HIST_IMAGE "..\\olympus\\pic.0746.jpg"
#define EXTENSION_IMAGE "..\\olympus\\pic.1070.jpg"

float sum_of_square_difference(const cv::Mat &src, const cv::Mat &dst);
float intersection_distance(const cv::Mat &m1, const cv::Mat &m2, bool is_3d);
int hist_normalize(cv::Mat &src, cv::Mat &dst, bool is_3d);
int rg_chrom_histogram(const cv::Mat &src, cv::Mat &dst);
int halves_rgb_histogram(const cv::Mat &src, cv::Mat &top, cv::Mat &bottom);
int texture_histogram(const cv::Mat &src, int *bins, float *normalized_bins);

int sobelX3x3(cv::Mat &src, cv::Mat &dst);
int sobelY3x3(cv::Mat &src, cv::Mat &dst);
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
#endif //PROJ2_INCLUDE_FEATURES_H_
