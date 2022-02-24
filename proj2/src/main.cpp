#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "features.h"
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <vector>

double deginrad(int degree) {
  return 2.0 * CV_PI / 360 * degree;
}

int pipeline(const std::string &src_dir,
             const std::vector<std::string> &files,
             std::vector<std::string> &top_n,
             int mode) {
  cv::Mat src = cv::imread(src_dir);
  cv::Mat dst(BINS, BINS, CV_32SC1, cv::Scalar(0));
  cv::Mat normalized_dst(BINS, BINS, CV_32F, cv::Scalar(0));

  int size[] = {BINS, BINS, BINS};
  cv::Mat top(3, size, CV_32SC1, cv::Scalar(0));
  cv::Mat bottom(3, size, CV_32SC1, cv::Scalar(0));
  cv::Mat normalized_top(3, size, CV_32F, cv::Scalar(0));
  cv::Mat normalized_bottom(3, size, CV_32F, cv::Scalar(0));

  cv::Mat greyscale_magnitude(src.rows, src.cols, CV_8UC1, cv::Scalar(0));
  int bins[BINS] = {0};
  int vertical_bins[BINS] = {0};
  float normalized_bins[BINS] = {0.f};
  float normalized_vertical_bins[BINS] = {0.f};

  // laws kernels
  float L5_arr[5] = {-1, -2, 0, 2, 1};
  cv::Mat L5(1, 5, CV_32F, L5_arr);
  cv::Mat E5_T(5, 1, CV_32F, L5_arr);
  cv::Mat kernel = E5_T * L5;
  cv::Mat src_laws_result(src.rows, src.cols, CV_8UC1, cv::Scalar(0));


  // Gabor kernels
  int kernel_size = 64;
  double sigma = 2.5, theta = 0, lambda = 5, gamma = 0.2, psi = 0;
  cv::Mat vertical_kernel = cv::getGaborKernel(cv::Size(kernel_size, kernel_size), sigma, theta, lambda, gamma, psi, CV_32F);
  cv::Mat horizontal_kernel = cv::getGaborKernel(cv::Size(kernel_size, kernel_size), sigma, deginrad(90), lambda, gamma, psi, CV_32F);
  cv::Mat normalized_gabor_dst = cv::Mat(BINS, BINS, CV_32F, cv::Scalar(0));

  // put all the files name as keys and distance score as values in a map
  std::vector<std::pair<std::string, float>> map;
  for (const std::string &file: files) {
    map.emplace_back(std::make_pair(file, 0.));
  }

  // for different tasks, different functions all called
  if (mode == 2) {
    rg_chrom_histogram(src, dst);
    hist_normalize(dst, normalized_dst, false);
  } else if (mode == 3) {
    halves_rgb_histogram(src, top, bottom);
    hist_normalize(top, normalized_top, true);
    hist_normalize(bottom, normalized_bottom, true);
  } else if (mode == 4) {
    // rg histogram
    rg_chrom_histogram(src, dst);
    hist_normalize(dst, normalized_dst, false);
    // calculate magnitude
    cv::Mat x_sobel_frame_16s(src.rows, src.cols, CV_16SC3, cv::Scalar(0));
    sobelX3x3(src, x_sobel_frame_16s);
    cv::Mat y_sobel_frame_16s(src.rows, src.cols, CV_16SC3, cv::Scalar(0));
    sobelY3x3(src, y_sobel_frame_16s);
    cv::Mat converted_frame(src.rows, src.cols, CV_8UC3, cv::Scalar(0));
    magnitude(x_sobel_frame_16s, y_sobel_frame_16s, converted_frame);
    // convert the magnitude image to greyscale image
    cv::cvtColor(converted_frame, greyscale_magnitude, cv::COLOR_BGR2GRAY);
    texture_histogram(greyscale_magnitude, bins, normalized_bins);
  } else if (mode == 5) {
    // rg chromaticity
    rg_chrom_histogram(src, dst);
    hist_normalize(dst, normalized_dst, false);

    // laws filter
    cv::Mat greyscale_src(src.rows, src.cols, CV_8UC1, cv::Scalar(0));
    cv::cvtColor(src.clone(), greyscale_src, cv::COLOR_BGR2GRAY);
    cv::Mat src_laws;
    cv::filter2D(greyscale_src, src_laws, CV_32F, kernel);
    cv::Mat mats[3];
    cv::split(src_laws, mats);
    for (int i = 0; i < mats[0].rows; i++) {
      for (int j = 0; j < mats[0].cols; j++) {
        float pixel = mats[0].at<float>(i, j);
        pixel = abs(pixel);
        src_laws_result.at<uchar>(i, j) = cvRound(fmin(pixel, 255));
      }
    }
    texture_histogram(src_laws_result, bins, normalized_bins);
  } else if (mode == 6) {
    cv::Mat greyscale_src(src.rows, src.cols, CV_8UC1);
    cv::cvtColor(src.clone(), greyscale_src, cv::COLOR_BGR2GRAY);

    // Gabor texture
    cv::Mat gabor_horizon_dst, gabor_vertical_dst;
    cv::filter2D(greyscale_src, gabor_horizon_dst, CV_32F, horizontal_kernel);
    cv::filter2D(greyscale_src, gabor_vertical_dst, CV_32F, vertical_kernel);

    double mins[4], maxs[4];

    minMaxIdx(gabor_horizon_dst, mins, maxs);
    cv::Mat src_horizon_gabor;
    gabor_horizon_dst.convertTo(src_horizon_gabor, CV_8UC1, 255.0 / (maxs[0] - mins[0]), -255 * mins[0] / (maxs[0] - mins[0]));

    minMaxIdx(gabor_vertical_dst, mins, maxs);
    cv::Mat src_vertical_gabor;
    gabor_vertical_dst.convertTo(src_vertical_gabor, CV_8UC1, 255.0 / (maxs[0] - mins[0]), -255 * mins[0] / (maxs[0] - mins[0]));

    texture_histogram(src_horizon_gabor, bins, normalized_bins);
    texture_histogram(src_vertical_gabor, vertical_bins, normalized_vertical_bins);
  }

  for (std::pair<std::string, float> &p: map) {
    cv::Mat img = cv::imread(p.first);
    // for each image in the database, perform the same operation as above and calculate the combined distance
    if (mode == 1) {
      p.second = sum_of_square_difference(src, img);
    } else if (mode == 2) {
      cv::Mat img_hist = cv::Mat(BINS, BINS, CV_32SC1, cv::Scalar(0));
      cv::Mat img_normalized_dst = cv::Mat(BINS, BINS, CV_32F, cv::Scalar(0));
      rg_chrom_histogram(img, img_hist);
      hist_normalize(img_hist, img_normalized_dst, false);
      p.second = intersection_distance(normalized_dst, img_normalized_dst, false);
    } else if (mode == 3) {
      cv::Mat img_top = cv::Mat(3, size, CV_32SC1, cv::Scalar(0));
      cv::Mat img_bottom = cv::Mat(3, size, CV_32SC1, cv::Scalar(0));
      cv::Mat img_normalized_top = cv::Mat(3, size, CV_32F, cv::Scalar(0));
      cv::Mat img_normalized_bottom = cv::Mat(3, size, CV_32F, cv::Scalar(0));
      halves_rgb_histogram(img, img_top, img_bottom);
      hist_normalize(img_top, img_normalized_top, true);
      hist_normalize(img_bottom, img_normalized_bottom, true);
      p.second = intersection_distance(normalized_top, img_normalized_top, true) * 0.5f
          + intersection_distance(normalized_bottom, img_normalized_bottom, true) * 0.5f;
    } else if (mode == 4) {
      cv::Mat img_dst = cv::Mat(BINS, BINS, CV_32SC1, cv::Scalar(0));
      cv::Mat img_normalized_dst = cv::Mat(BINS, BINS, CV_32F, cv::Scalar(0));
      rg_chrom_histogram(img, img_dst);
      hist_normalize(img_dst, img_normalized_dst, false);

      cv::Mat img_greyscale_magnitude(img.rows, img.cols, CV_8UC1);
      cv::Mat img_x_sobel_frame_16s(img.rows, img.cols, CV_16SC3, cv::Scalar(0));
      sobelX3x3(img, img_x_sobel_frame_16s);
      cv::Mat img_y_sobel_frame_16s(img.rows, img.cols, CV_16SC3, cv::Scalar(0));
      sobelY3x3(img, img_y_sobel_frame_16s);
      cv::Mat img_converted_frame(img.rows, img.cols, CV_8UC3, cv::Scalar(0));
      magnitude(img_x_sobel_frame_16s, img_y_sobel_frame_16s, img_converted_frame);
      cv::cvtColor(img_converted_frame, img_greyscale_magnitude, cv::COLOR_BGR2GRAY);

      int img_bins[BINS] = {0};
      float img_normalized_bins[BINS] = {0.f};
      texture_histogram(img_greyscale_magnitude, img_bins, img_normalized_bins);

      float texture_score = 0.f;
      for (int i = 0; i < BINS; i++) {
        texture_score += std::min(normalized_bins[i], img_normalized_bins[i]);
      }
      p.second = texture_score * 0.5f + intersection_distance(normalized_dst, img_normalized_dst, false) * 0.5f;
    } else if (mode == 5) {
      cv::Mat img_dst = cv::Mat(BINS, BINS, CV_32SC1, cv::Scalar(0));
      cv::Mat img_normalized_dst = cv::Mat(BINS, BINS, CV_32F, cv::Scalar(0));
      rg_chrom_histogram(img, img_dst);
      hist_normalize(img_dst, img_normalized_dst, false);

      cv::Mat img_greyscale(img.rows, img.cols, CV_8UC1);
      cv::cvtColor(img.clone(), img_greyscale, cv::COLOR_BGR2GRAY);
      cv::Mat img_laws;
      cv::filter2D(img_greyscale, img_laws, CV_32F, kernel);
      cv::Mat mats[3];
      cv::split(img_laws, mats);
      cv::Mat img_laws_result(src.rows, src.cols, CV_8UC1, cv::Scalar(0));
      for (int i = 0; i < mats[0].rows; i++) {
        for (int j = 0; j < mats[0].cols; j++) {
          float pixel = mats[0].at<float>(i, j);
          pixel = abs(pixel);
          pixel = fmin(pixel, 255);
          img_laws_result.at<uchar>(i, j) = pixel;
        }
      }
      int img_bins[BINS] = {0};
      float img_normalized_bins[BINS] = {0.f};
      texture_histogram(img_laws_result, img_bins, img_normalized_bins);
      float texture_score = 0.f;
      for (int i = 0; i < BINS; i++) {
        texture_score += std::min(normalized_bins[i], img_normalized_bins[i]);
      }
      float color_score = intersection_distance(normalized_dst, img_normalized_dst, false);
      p.second = texture_score * 0.3f + color_score * 0.7f;
    } else if (mode == 6) {
      cv::Mat greyscale_img(img.rows, img.cols, CV_8UC1);
      cv::cvtColor(img.clone(), greyscale_img, cv::COLOR_BGR2GRAY);

      cv::Mat img_horizon_dst, img_vertical_dst;
      cv::filter2D(greyscale_img, img_horizon_dst, CV_32F, horizontal_kernel);
      cv::filter2D(greyscale_img, img_vertical_dst, CV_32F, vertical_kernel);
      double mins[4], maxs[4];
      minMaxIdx(img_horizon_dst, mins, maxs);
      cv::Mat img_horizon_gabor;
      img_horizon_dst.convertTo(img_horizon_gabor, CV_8UC1, 255.0 / (maxs[0] - mins[0]), -255 * mins[0] / (maxs[0] - mins[0]));
      minMaxIdx(img_vertical_dst, mins, maxs);
      cv::Mat img_vertical_gabor;
      img_vertical_dst.convertTo(img_vertical_gabor, CV_8UC1, 255.0 / (maxs[0] - mins[0]), -255 * mins[0] / (maxs[0] - mins[0]));

      int img_horizon_bins[BINS] = {0};
      float img_normalized_horizon_bins[BINS] = {0.f};
      int img_vertical_bins[BINS] = {0};
      float img_normalized_vertical_bins[BINS] = {0.f};
      texture_histogram(img_horizon_gabor, img_horizon_bins, img_normalized_horizon_bins);
      texture_histogram(img_vertical_gabor, img_vertical_bins, img_normalized_vertical_bins);


      float horizon_texture_score = 0.f;
      float vertical_texture_score = 0.f;
      for (int i = 0; i < BINS; i++) {
        horizon_texture_score += std::min(normalized_bins[i], img_normalized_horizon_bins[i]);
        vertical_texture_score += std::min(normalized_vertical_bins[i], img_normalized_vertical_bins[i]);
      }
      p.second = horizon_texture_score * 0.5f + vertical_texture_score * 0.5f;
    }
  }
  // sort the map<img_name, distance> by the value
  if (mode == 1) {
    sort(map.begin(), map.end(), [=](std::pair<std::string, float> &a, std::pair<std::string, float> &b) {
      return a.second < b.second;
    });
  } else {
    sort(map.begin(), map.end(), [=](std::pair<std::string, float> &a, std::pair<std::string, float> &b) {
      return a.second > b.second;
    });
  }


  if (mode != 5) {
    // assert the first match image is itself
    assert(map.at(0).first == src_dir);
    // return the top three matching images except itself
    top_n.push_back(map.at(1).first);
    top_n.push_back(map.at(2).first);
    top_n.push_back(map.at(3).first);
    std::cout << "The top 3 matches for task 5 are:" << std::endl;
    for (int i = 1; i < 4; i++) {
      std::cout << map.at(i).first << std::endl;
    }
  } else {
    std::cout << "The top 10 matches for task 5 are:" << std::endl;
    // print the top 10 matches to the terminal
    for (int i = 1; i < 11; i++) {
      std::cout << map.at(i).first << std::endl;
    }
    top_n.push_back(map.at(1).first);
    top_n.push_back(map.at(2).first);
    top_n.push_back(map.at(3).first);
  }
  return 0;
}

int read_files(char *img_dir, std::vector<std::string> &files) {
  char dirname[256];
  char buffer[256];
  DIR *dirp;
  struct dirent *dp;

  strcpy(dirname, img_dir);
  printf("Processing directory %s\n", dirname);

  // open the directory
  dirp = opendir(dirname);
  if (dirp == nullptr) {
    printf("Cannot open directory %s\n", dirname);
    exit(-1);
  }

  // loop over all the files in the image file listing
  while ((dp = readdir(dirp)) != nullptr) {

    // check if the file is an image
    if (strstr(dp->d_name, ".jpg") ||
        strstr(dp->d_name, ".png") ||
        strstr(dp->d_name, ".ppm") ||
        strstr(dp->d_name, ".tif")) {

      // build the overall filename
      strcpy(buffer, dirname);
      // change this line since the delimiter of file path in windows os is '\' instead of '/'
      strcat(buffer, "\\");
      strcat(buffer, dp->d_name);

      files.emplace_back(buffer);
    }
  }
  return 0;
}

int main(int argc, char *argv[]) {
  std::vector<std::string> files;
  int rows = 512, cols = 640;
  cv::Mat src;
  std::vector<std::string> top_n;

  // check for sufficient arguments
  if (argc < 3) {
    printf("usage: %s <directory path> <task>\n", argv[0]);
    exit(-1);
  }

  read_files(argv[1], files);

  char *mode_arg = argv[2];
  if (strlen(mode_arg) > 1) {
    printf("Mode should be one of the number from 1 to 6");
    exit(-1);
  }

  int mode = mode_arg[0] - '0';
  switch (mode) {
    case 1: {
      pipeline(BASELINE_IMAGE, files, top_n, mode);
      break;
    }
    case 2: {
      pipeline(COLOR_HIST_IMAGE, files, top_n, mode);
      break;
    }
    case 3: {
      pipeline(MULTI_HIST_IMAGE, files, top_n, mode);
      break;
    }
    case 4: {
      pipeline(TEXTURE_COLOR_HIST_IMAGE, files, top_n, mode);
      break;
    }
    case 5: {
      pipeline(CUSTOM_HIST_IMAGE, files, top_n, mode);
      break;
    }
    case 6: {
      pipeline(EXTENSION_IMAGE, files, top_n, mode);
      break;
    }
    default: {
      printf("Mode should be one of the number from 1 to 6");
      exit(-1);
    }
  }

  // the top three matches in the window
  cv::Mat dst = cv::Mat(rows, 3 * cols, CV_8UC3);
  cv::imread(top_n.at(0)).copyTo(dst.rowRange(0, rows).colRange(0, cols));
  cv::imread(top_n.at(1)).copyTo(dst.rowRange(0, rows).colRange(cols, 2 * cols));
  cv::imread(top_n.at(2)).copyTo(dst.rowRange(0, rows).colRange(2 * cols, 3 * cols));
  cv::namedWindow("window");
  cv::imshow("window", dst);
  while (true) {
    int key = cv::waitKey(10);
    if (key == 'q') {
      break;
    }
    if (key == 's') {
      cv::imwrite("..\\result.jpg", dst);
    }
  }
  return 0;
}

