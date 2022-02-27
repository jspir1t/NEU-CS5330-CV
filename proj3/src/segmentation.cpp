#include "segmentation.h"

int threshold(const cv::Mat &src, cv::Mat &dst, int threshold) {
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
      if ((int) pixel[0] < threshold && (int) pixel[1] < threshold && (int) pixel[2] < threshold) {
        dst.at<uchar>(i, j) = FRONT_GROUND;
      } else {
        dst.at<uchar>(i, j) = BACK_GROUND;
      }
    }
  }
  return 0;
}

int grass_fire_transform(const cv::Mat &src, cv::Mat &dst, int FRONT_END_VALUE) {
  // Pass 1
  int up, left;
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      if (src.at<uchar>(i, j) == FRONT_END_VALUE) {
        up = i == 0 ? 0 : dst.at<int>(i - 1, j);
        left = j == 0 ? 0 : dst.at<int>(i, j - 1);
        dst.at<int>(i, j) = 1 + std::min(up, left);
      } else {
        dst.at<int>(i, j) = 0;
      }
    }
  }

  // Pass 2
  int down, right;
  for (int i = src.rows - 1; i >= 0; i--) {
    for (int j = src.cols - 1; j >= 0; j--) {
      if (src.at<uchar>(i, j) == FRONT_END_VALUE) {
        down = i == src.rows - 1 ? 0 : dst.at<int>(i + 1, j);
        right = j == src.cols - 1 ? 0 : dst.at<int>(i, j + 1);
        dst.at<int>(i, j) = std::min(dst.at<int>(i, j), 1 + std::min(down, right));
      } else {
        dst.at<int>(i, j) = 0;
      }
    }
  }
  return 0;
}

int cleanup(cv::Mat &src, cv::Mat &dst, int steps) {
  // shrink at first
  cv::Mat src_grass_file(src.rows, src.cols, CV_32S, cv::Scalar(0));
  grass_fire_transform(src, src_grass_file, FRONT_GROUND);

  cv::Mat temp(src.rows, src.cols, CV_8UC1, cv::Scalar(0));
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      if (src_grass_file.at<int>(i, j) <= steps) {
        temp.at<uchar>(i, j) = BACK_GROUND;
      } else {
        temp.at<uchar>(i, j) = src.at<uchar>(i, j);
      }
    }
  }

  // grow steps
  temp.copyTo(dst);
  // since we shrink steps above, grow back steps
  for (int k = 0; k < steps; k++) {
    for (int i = 0; i < dst.rows; i++) {
      for (int j = 0; j < dst.cols; j++) {
        if (temp.at<uchar>(i, j) == BACK_GROUND) {
          bool left = (i > 0) && (temp.at<uchar>(i - 1, j) == FRONT_GROUND);
          bool right = (i < temp.rows - 1) && (temp.at<uchar>(i+1, j) == FRONT_GROUND);
          bool up = (j > 0) && (temp.at<uchar>(i, j-1) == FRONT_GROUND);
          bool down = (j < temp.cols - 1) && (temp.at<uchar>(i, j+1) == FRONT_GROUND);
          if (left || right || up || down) {
            dst.at<uchar>(i, j) = FRONT_GROUND;
          }
        }
      }
    }
    dst.copyTo(temp);
  }
  return 0;
}

// the components do not include background
int segment(cv::Mat &src, cv::Mat &dst, int component_num, std::map<int, cv::Mat> &regions, int min_area, cv::Mat &major) {
  // labels for each component
  std::set<int> labels;
  cv::Mat label_img(src.size(), CV_32S);
  cv::Mat stats, centroids;
  // get the number of components including the background
  int n_labels = cv::connectedComponentsWithStats(src, label_img, stats, centroids, 8);
  // if only background in the image, return
  if (n_labels <= 1) {
    return -1;
  }

  // find components that are adjacent to the border
  std::set<int> border_components;
  for (int i = 0; i < label_img.rows; i++) {
    for (int j = 0; j < label_img.cols; j++) {
      if (i == 0 || i == label_img.rows - 1 || j == 0 || j == label_img.cols - 1) {
        border_components.insert(label_img.at<int>(i, j));
      }
    }
  }

  // store the components with its label and size of area(skip background)
  std::vector<std::pair<int, int>> components;
  for (int i = 1; i < n_labels; i++) {
    // skip those components that are adjacent to the border
    if (!border_components.count(i)) {
      int area = stats.at<int>(i, cv::CC_STAT_AREA);
      // skip those components whose area is less than min_area
      if (area >= min_area) {
        components.emplace_back(i, stats.at<int>(i, cv::CC_STAT_AREA));
      }
    }
  }
  std::sort(components.begin(),
            components.end(),
            [](const std::pair<int, int> &left, const std::pair<int, int> &right) {
              return left.second > right.second;
            });
  if (components.empty()) {
    return -1;
  }

  // colors for drawing different components
  std::vector<cv::Vec3b> colors(n_labels);
  // background color
  colors[0] = cv::Vec3b(0, 0, 0);
  srand(1);
  for (int i = 1; i < n_labels; i++) {
    colors[i] = cv::Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
  }

  int size = std::min((int)(components.size()), component_num);
  // store the labels the top k components regardless of background or not
  for (int i = 0; i < size; i++) {
    labels.insert(components[i].first);
  }
  for (int i = 0; i < dst.rows; ++i) {
    for (int j = 0; j < dst.cols; ++j) {
      // draw the top k components
      int label = label_img.at<int>(i, j);
      // if the pixel belongs to the top K components
      if (labels.count(label)) {
        dst.at<cv::Vec3b>(i, j) = colors[label];
        // create the mat if it is not in the map yet
        if (!regions.count(label)) {
          regions.emplace(std::make_pair(label, cv::Mat(src.rows, src.cols, CV_8UC1, cv::Scalar(0))));
        }
        // set the pixel in the corresponding mat as 255
        regions[label].at<uchar>(i, j) = 255;
      }
    }
  }
  // the major component should be the one with largest area
  major = regions[components[0].first];
  return 0;
}

void mark_object(cv::Mat &segmented_img, std::vector<cv::Point> draw_vertices) {
  cv::circle(segmented_img, draw_vertices[0], 2, cv::Scalar(0, 255, 255), 2);
  cv::line(segmented_img, draw_vertices[1], draw_vertices[2], cv::Scalar(0, 255, 0), 2);
  cv::line(segmented_img, draw_vertices[3], draw_vertices[4], cv::Scalar(0, 255, 0), 2);
  cv::line(segmented_img, draw_vertices[4], draw_vertices[6], cv::Scalar(0, 255, 0), 2);
  cv::line(segmented_img, draw_vertices[6], draw_vertices[5], cv::Scalar(0, 255, 0), 2);
  cv::line(segmented_img, draw_vertices[5], draw_vertices[3], cv::Scalar(0, 255, 0), 2);
}

int features(cv::Mat &src, std::vector<double> &feature_vector, std::vector<cv::Point> &draw_vertices) {
  cv::Moments moments = cv::moments(src, true);

  // centroid
  std::pair<double, double> centroid;
  centroid.first = moments.m10 / moments.m00;
  centroid.second = moments.m01 / moments.m00;
  draw_vertices.emplace_back(cv::Point((int)centroid.first, (int)centroid.second));

  // angle
  double angle = 0.5 * std::atan2(2.0 * moments.mu11, moments.mu20 - moments.mu02);

  // coordinate system transformation
  std::vector<std::pair<double, double>> transformed_vertices;
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      if (src.at<uchar>(i, j) == FRONT_GROUND) {
        // when transforming, the x is j and y is i
        int x = j, y = i;
        transformed_vertices.emplace_back(std::make_pair(
            std::cos(angle) * (x - centroid.first) + std::sin(angle) * (y - centroid.second),
            std::cos(angle) * (y - centroid.second) - std::sin(angle) * (x - centroid.first)
        ));
      }
    }
  }

  // get bounding boxes axis
  double quad_axis[4] = {0., 0., 0., 0.};
  for (std::pair<double, double> p: transformed_vertices) {
    // leftest x-axis
    quad_axis[0] = std::min(p.first, quad_axis[0]);
    // rightest x-axis
    quad_axis[1] = std::max(p.first, quad_axis[1]);
    // top y-axis
    quad_axis[2] = std::max(p.second, quad_axis[2]);
    // bottom y-axis
    quad_axis[3] = std::min(p.second, quad_axis[3]);
  }

  // get the width and height feature
  double height = quad_axis[2] - quad_axis[3];
  double width = quad_axis[1] - quad_axis[0];
  feature_vector.emplace_back(height / width);


  // get filled ratio feature
  feature_vector.emplace_back(moments.m00 / (height * width));

  // get second moment about the central axis feature
  double beta = angle + CV_PI / 2.;
  double summation = 0.;
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      if (src.at<uchar>(i, j) == FRONT_GROUND) {
        int x = j, y = i;
        double temp = (y - centroid.second) * std::cos(beta) + (x - centroid.first) * std::sin(beta);
        summation += temp * temp;
      }
    }
  }
  double mu_22_angle = summation / moments.m00;
  feature_vector.emplace_back(mu_22_angle);

  double hu[7];
  HuMoments(moments, hu);
  feature_vector.emplace_back(hu[0]);
  feature_vector.emplace_back(hu[1]);
  feature_vector.emplace_back(hu[2]);

  // find the coordinates in the original coordinate system, then draw the bounding box
  std::vector<std::pair<double, double>> temp_vertices;
  // central axis vertices
  temp_vertices.emplace_back(std::make_pair(quad_axis[0], 0));
  temp_vertices.emplace_back(std::make_pair(quad_axis[1], 0));
  // top left vertices
  temp_vertices.emplace_back(std::make_pair(quad_axis[0], quad_axis[2]));
  // top right vertices
  temp_vertices.emplace_back(std::make_pair(quad_axis[1], quad_axis[2]));
  // bottom left vertices
  temp_vertices.emplace_back(std::make_pair(quad_axis[0], quad_axis[3]));
  // bottom right vertices
  temp_vertices.emplace_back(std::make_pair(quad_axis[1], quad_axis[3]));

  for (std::pair<int, int> p: temp_vertices) {
    draw_vertices.emplace_back(cv::Point(
        (int)(std::cos(angle) * p.first - std::sin(angle) * p.second + centroid.first),
        (int)(std::sin(angle) * p.first + std::cos(angle) * p.second + centroid.second)
    ));
  }
  return 0;
}