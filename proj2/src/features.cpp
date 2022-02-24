#include "features.h"

float sum_of_square_difference(const cv::Mat &src, const cv::Mat &dst) {
  int mid_row = dst.rows / 2;
  int mid_col = dst.cols / 2;
  float sums[] = {0.f, 0.f, 0.f};
  for (int i = mid_row - 4; i <= mid_row + 4; i++) {
    for (int j = mid_col - 4; j <= mid_col + 4; j++) {
      for (int c = 0; c < dst.channels(); c++) {
        int diff = dst.at<cv::Vec3b>(i, j)[c] - src.at<cv::Vec3b>(i, j)[c];
        sums[c] += (float)(diff * diff);
      }
    }
  }
  return sums[0] + sums[1] + sums[2];
}

int hist_normalize(cv::Mat &src, cv::Mat &dst, bool is_3d) {
  long total_value = 0l;
  if (is_3d) {
    for (int i = 0; i < src.size[0]; i++) {
      for (int j = 0; j < src.size[1]; j++) {
        for (int k = 0; k < src.size[2]; k++) {
          total_value += src.at<int>(i, j, k);
        }
      }
    }
    if (total_value == 0l)
      return 0;
    for (int i = 0; i < dst.size[0]; i++) {
      for (int j = 0; j < dst.size[1]; j++) {
        for (int k = 0; k < src.size[2]; k++) {
          float percent = (float)src.at<int>(i, j, k) / total_value;
          dst.at<float>(i, j, k) = percent;
        }
      }
    }
  } else {
    for (int i = 0; i < src.rows; i++) {
      for (int j = 0; j < src.cols; j++) {
        total_value += src.at<int>(i, j);
      }
    }
    if (total_value == 0)
      return 0;
    for (int i = 0; i < dst.rows; i++) {
      for (int j = 0; j < dst.cols; j++) {
        float percent = (float)src.at<int>(i, j) / total_value;
        dst.at<float>(i, j) = percent;
      }
    }
  }
  return 0;
}

float intersection_distance(const cv::Mat &m1, const cv::Mat &m2, bool is_3d) {
  float intersection = 0.f;
  if (is_3d) {
    for (int i = 0; i < m1.size[0]; i++) {
      for (int j = 0; j < m1.size[1]; j++) {
        for (int k = 0; k < m1.size[2]; k++) {
          intersection += std::min(m1.at<float>(i, j, k), m2.at<float>(i, j, k));
        }
      }
    }
  } else {
    for (int i = 0; i < m1.rows; i++) {
      for (int j = 0; j < m1.cols; j++) {
        intersection += std::min(m1.at<float>(i, j), m2.at<float>(i, j));
      }
    }
  }
  return intersection;
}

int rg_chrom_histogram(const cv::Mat &src, cv::Mat &dst) {
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      // add one to the sum of rgb channel values, avoid dividing-by-zero and out-of-bound problem
      float pixel_sum = (float)src.at<cv::Vec3b>(i, j)[0] + (float)src.at<cv::Vec3b>(i, j)[1] + (float)src.at<cv::Vec3b>(i, j)[2] + 1;
      int g_denominator = src.at<cv::Vec3b>(i, j)[1] * BINS;
      int r_denominator = src.at<cv::Vec3b>(i, j)[2] * BINS;
      float g_index = (float)g_denominator / pixel_sum;
      float r_index = (float)r_denominator / pixel_sum;
      dst.at<int>((int)g_index, (int)r_index) += 1;
    }
  }
  return 0;
}

int halves_rgb_histogram(const cv::Mat &src, cv::Mat &top, cv::Mat &bottom) {
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      int indices[] = {0, 0, 0};
      for (int c = 0; c < src.channels(); c++) {
        int denominator = src.at<cv::Vec3b>(i, j)[c] * BINS;
        indices[c] = denominator / 256;
      }
      if (i < src.rows / 2) {
        top.at<int>(indices[0], indices[1], indices[2]) += 1;
      } else {
        bottom.at<int>(indices[0], indices[1], indices[2]) += 1;
      }
    }
  }
  return 0;
}

int texture_histogram(const cv::Mat &src, int *bins, float *normalized_bins) {
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      for (int c = 0; c < src.channels(); c++) {
        int denominator = src.at<cv::Vec3b>(i, j)[c] * BINS;
        bins[denominator / 256] += 1;
      }
    }
  }
  int total_value = 1;
  for (int i = 0; i < BINS; i++) {
    total_value += bins[i];
  }
  for (int i = 0; i < BINS; i++) {
    normalized_bins[i] = (float)bins[i] / total_value;
  }
  return 0;
}

int separable_sobel(cv::Mat &src, cv::Mat &dst, const float horizontal_kernel[3], const float vertical_kernel[3]) {
  cv::Mat converted(src.rows, src.cols, CV_16SC3);

  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      converted.at<short>(i, j) = (short)(src.at<uchar>(i, j));
      dst.at<short>(i, j) = (short)(src.at<uchar>(i, j));
    }
  }

  // horizontal
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j <= src.cols - 3; j++) {
      float b_sum = 0.0, g_sum = 0.0, r_sum = 0.0;
      for (int k = 0; k < 3; k++) {
        b_sum += src.at<cv::Vec3b>(i, j + k)[0] * horizontal_kernel[k];
        g_sum += src.at<cv::Vec3b>(i, j + k)[1] * horizontal_kernel[k];
        r_sum += src.at<cv::Vec3b>(i, j + k)[2] * horizontal_kernel[k];
      }
      converted.at<cv::Vec3s>(i, j + 1) = cv::Vec3s(b_sum, g_sum, r_sum);
    }
  }

  // vertical
  for (int i = 0; i <= src.rows - 3; i++) {
    for (int j = 0; j <= src.cols; j++) {
      float b_sum = 0.0, g_sum = 0.0, r_sum = 0.0;
      for (int k = 0; k < 3; k++) {
        b_sum += converted.at<cv::Vec3s>(i + k, j)[0] * vertical_kernel[k];
        g_sum += converted.at<cv::Vec3s>(i + k, j)[1] * vertical_kernel[k];
        r_sum += converted.at<cv::Vec3s>(i + k, j)[2] * vertical_kernel[k];
      }
      dst.at<cv::Vec3s>(i + 1, j) = cv::Vec3s(b_sum, g_sum, r_sum);
    }
  }
  return 0;
}

int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
  float horizontal_kernel[3] = {1.0f, 0.0f, -1.0f};
  float vertical_kernel[3] = {0.25f, 0.5f, 0.25f};
  separable_sobel(src, dst, horizontal_kernel, vertical_kernel);
  return 0;
}

int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
  float horizontal_kernel[3] = {0.25f, 0.5f, 0.25f};
  float vertical_kernel[3] = {1.0f, 0.0f, -1.0f};
  separable_sobel(src, dst, horizontal_kernel, vertical_kernel);
  return 0;
}

int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {
  for (int i = 0; i < sx.rows; i++) {
    for (int j = 0; j < sx.cols; j++) {
      cv::Vec3s sx_pixel = sx.at<cv::Vec3s>(i, j);
      cv::Vec3s sy_pixel = sy.at<cv::Vec3s>(i, j);
      dst.at<cv::Vec3b>(i, j) = cv::Vec3b(
          sqrt(sx_pixel[0] * sx_pixel[0] + sy_pixel[0] * sy_pixel[0]),
          sqrt(sx_pixel[1] * sx_pixel[1] + sy_pixel[1] * sy_pixel[1]),
          sqrt(sx_pixel[2] * sx_pixel[2] + sy_pixel[2] * sy_pixel[2]));
    }
  }
  return 0;
}