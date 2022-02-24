#include "../include/filter.h"
#include<cmath>

int greyscale(cv::Mat &src, cv::Mat &dst) {
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
      // copy green color channel to the other two channels
      dst.at<cv::Vec3b>(i, j) = cv::Vec3b(pixel[1], pixel[1], pixel[1]);
    }
  }
  return 0;
}

int blur5x5(cv::Mat &src, cv::Mat &dst) {
  int kernel[] = {1, 2, 4, 2, 1};
  cv::Mat converted;
  // copy the src Mat to the converted and dst mat, which could make sure the values in the border pixel are not zero
  src.copyTo(converted);
  src.copyTo(dst);

  // compute the product of the kernel and the area under it
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j <= src.cols - 5; j++) {
      int b_sum = 0, g_sum = 0, r_sum = 0;
      for (int k = 0; k < 5; k++) {
        b_sum += src.at<cv::Vec3b>(i, j + k)[0] * kernel[k];
        g_sum += src.at<cv::Vec3b>(i, j + k)[1] * kernel[k];
        r_sum += src.at<cv::Vec3b>(i, j + k)[2] * kernel[k];
      }
      converted.at<cv::Vec3b>(i, j + 2) =
          // divide by 10 to narrow it to unsigned char
          cv::Vec3b(b_sum / 10, g_sum / 10, r_sum / 10);
    }
  }

  // compute the product of the kernel transpose and the area under it
  for (int i = 0; i <= src.rows - 5; i++) {
    for (int j = 0; j < src.cols; j++) {
      int b_sum = 0, g_sum = 0, r_sum = 0;
      for (int k = 0; k < 5; k++) {
        b_sum += converted.at<cv::Vec3b>(i + k, j)[0] * kernel[k];
        g_sum += converted.at<cv::Vec3b>(i + k, j)[1] * kernel[k];
        r_sum += converted.at<cv::Vec3b>(i + k, j)[2] * kernel[k];
      }
      dst.at<cv::Vec3b>(i + 2, j) = cv::Vec3b(b_sum / 10, g_sum / 10, r_sum / 10);
    }
  }
  return 0;
}

int separable_sobel(cv::Mat &src, cv::Mat &dst, const float horizontal_kernel[3], const float vertical_kernel[3]) {
  cv::Mat converted(src.rows, src.cols, CV_16SC3);

  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      converted.at<cv::Vec3s>(i, j) = cv::Vec3s(src.at<cv::Vec3b>(i, j));
      dst.at<cv::Vec3s>(i, j) = cv::Vec3s(src.at<cv::Vec3b>(i, j));
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

int transform(cv::Mat &src, cv::Mat &dst) {
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      cv::Vec3s pixel = src.at<cv::Vec3s>(i, j);
      // For displaying, we convert the type of value from Vec3s to Vec3b by keeping the absolute value
      dst.at<cv::Vec3b>(i, j) = cv::Vec3b(abs(pixel[0]), abs(pixel[1]), abs(pixel[2]));
    }
  }
  return 0;
}

int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
  float horizontal_kernel[3] = {1.0, 0.0, -1.0};
  float vertical_kernel[3] = {0.25, 0.5, 0.25};
  separable_sobel(src, dst, horizontal_kernel, vertical_kernel);
  return 0;
}

int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
  float horizontal_kernel[3] = {0.25, 0.5, 0.25};
  float vertical_kernel[3] = {1.0, 0.0, -1.0};
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

int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels) {
  blur5x5(src, dst);
  int b = 255 / levels;
  for (int i = 0; i < dst.rows; i++) {
    for (int j = 0; j < dst.cols; j++) {
      dst.at<cv::Vec3b>(i, j)[0] /= b;
      dst.at<cv::Vec3b>(i, j)[0] *= b;
      dst.at<cv::Vec3b>(i, j)[1] /= b;
      dst.at<cv::Vec3b>(i, j)[1] *= b;
      dst.at<cv::Vec3b>(i, j)[2] /= b;
      dst.at<cv::Vec3b>(i, j)[2] *= b;
    }
  }
  return 0;
}

int cartoon(cv::Mat &src, cv::Mat &dst, int levels, int magThreshold) {
  cv::Mat x_sobel_result(src.rows, src.cols, CV_16SC3);
  sobelX3x3(src, x_sobel_result);
  cv::Mat y_sobel_result(src.rows, src.cols, CV_16SC3);
  sobelY3x3(src, y_sobel_result);
  cv::Mat magnitude_result(src.rows, src.cols, CV_8UC3);
  magnitude(x_sobel_result, y_sobel_result, magnitude_result);
  blurQuantize(src, dst, levels);
  for (int i = 0; i < dst.rows; i++) {
    for (int j = 0; j < dst.cols; j++) {
      for (int channel = 0; channel < 3; channel++) {
        if (magnitude_result.at<cv::Vec3b>(i, j)[channel] > magThreshold) {
          dst.at<cv::Vec3b>(i, j)[channel] = 0;
        }
      }
    }
  }
  return 0;
}

int negative(cv::Mat &src, cv::Mat &dst) {
  for (int i = 0; i < dst.rows; i++) {
    for (int j = 0; j < dst.cols; j++) {
      dst.at<cv::Vec3b>(i, j) = cv::Vec3b(255 - src.at<cv::Vec3b>(i, j)[0],
                                          255 - src.at<cv::Vec3b>(i, j)[1],
                                          255 - src.at<cv::Vec3b>(i, j)[2]);
    }
  }
  return 0;
}

// Sepia is not a filter operation, it is a weighted sum of the whole three channels
int sepia(cv::Mat &src, cv::Mat &dst) {
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      double b_sum, g_sum, r_sum;
      b_sum =
          src.at<cv::Vec3b>(i, j)[2] * 0.272 + src.at<cv::Vec3b>(i, j)[1] * 0.534 + src.at<cv::Vec3b>(i, j)[0] * 0.131;
      g_sum =
          src.at<cv::Vec3b>(i, j)[2] * 0.349 + src.at<cv::Vec3b>(i, j)[1] * 0.686 + src.at<cv::Vec3b>(i, j)[0] * 0.168;
      r_sum =
          src.at<cv::Vec3b>(i, j)[2] * 0.393 + src.at<cv::Vec3b>(i, j)[1] * 0.769 + src.at<cv::Vec3b>(i, j)[0] * 0.189;
      b_sum = fmin(255, b_sum);
      g_sum = fmin(255, g_sum);
      r_sum = fmin(255, r_sum);
      dst.at<cv::Vec3b>(i, j) = cv::Vec3b(b_sum, g_sum, r_sum);
    }
  }
  return 0;
}

int emboss(cv::Mat &src, cv::Mat &dst) {
  cv::Mat converted(src.rows, src.cols, CV_8UC3);
  // do the greyscale for this 3-channels mat at first
  greyscale(src, converted);
  double kernel[3][3] = {{-2, -1, 0},
                         {-1, 1, 1},
                         {0, 1, 2}};
  double b_sum = 0, g_sum = 0, r_sum = 0;
  // filtering
  for (int i = 1; i < converted.rows-1; i++) {
    for (int j = 1; j < converted.cols-1; j++) {
      b_sum = 0;
      g_sum = 0;
      r_sum = 0;
      for (int dx = -1; dx < 2; dx++) {
        for (int dy = -1; dy < 2; dy++) {
          int new_row = i + dx, new_col = j + dy;
            b_sum += converted.at<cv::Vec3b>(new_row, new_col)[0] * kernel[1+dx][1+dy];
            g_sum += converted.at<cv::Vec3b>(new_row, new_col)[1] * kernel[1+dx][1+dy];
            r_sum += converted.at<cv::Vec3b>(new_row, new_col)[2] * kernel[1+dx][1+dy];
        }
      }
      // since the coefficients in the kernel could be negative, I narrow the final value into [0, 255]
      b_sum = fmax(b_sum, 0);
      b_sum = fmin(b_sum, 255);
      g_sum = fmax(g_sum, 0);
      g_sum = fmin(g_sum, 255);
      r_sum = fmax(r_sum, 0);
      r_sum = fmin(r_sum, 255);
      dst.at<cv::Vec3b>(i, j) = cv::Vec3b((uchar) b_sum, (uchar) g_sum, (uchar) r_sum);
    }
  }
  return 0;
}