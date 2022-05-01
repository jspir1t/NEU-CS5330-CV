/**
 * utils file consists of the detail implementation of the QR code detection and the CSV file writer.
 */
#include "utils.h"
#include <fstream>
using namespace std;

void writeToCSV(const map<String, bool> &map) {
  String csvFile = "../results/self.csv";
  // clear content if exists
  std::ofstream f(csvFile, ofstream::out | ofstream::trunc);
  if (f.good()) {
    f.open(csvFile, ofstream::out | ofstream::trunc);
  }
  f.close();

  fstream csv;
  csv.open(csvFile, std::ios_base::app);
  // write the header into csv file
  csv << "algorithm" << ",";
  for (const auto &elem: map) {
    csv << elem.first.substr(0, elem.first.size() - 4) << ",";
  }
  csv << "\n";
  // write the corresponding values into csv file
  csv << "self" << ",";
  for (const auto &elem: map) {
    csv << elem.second << ",";
  }
  csv.close();
}


void drawBBox(Mat &image, vector<Point> &pts) {
  // find the outer rectangle that covers all those boxes vertices collected above
  RotatedRect rrt = minAreaRect(pts);
  Point2f vertices[4];
  // copy the four points into vertices array
  rrt.points(vertices);
  // draw the bounding box based on the the four vertices
  for (int i = 0; i < 4; i++) {
    line(image, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 2);
  }
}

vector<Point> detectAndDecode(Mat &image, bool showWin) {
  Mat gray, binary;
  Mat allContours = image.clone();
  cvtColor(image, gray, COLOR_BGR2GRAY);
  threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
  // in evaluation mode, the showWin should be false to avoid displaying many windows
  if (showWin) {
    imshow("binary", binary);
  }

  // detect contours
  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  findContours(binary.clone(), contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE, Point());

  Mat binaryPositionBoxesMat = Mat::zeros(image.size(), CV_8UC1);
  int positionBoxCnt = 0;
  for (size_t t = 0; t < contours.size(); t++) {
    double area = contourArea(contours[t]);
    // filter small size contour
    if (area < 100) continue;

    // Finds a rotated rectangle of the minimum area enclosing the contour
    RotatedRect rect = minAreaRect(contours[t]);
    float width = rect.size.width;
    float height = rect.size.height;
    float rate = min(width, height) / max(width, height);
    // if the rotated rectangle has rate bigger than a threshold and small enough(smaller than quarter of the cols and rows)
    if (rate > 0.7 && width < image.cols / 3 && height < image.rows / 3) {
      drawContours(allContours, contours, static_cast<int>(t), Scalar(255, 0, 0), 2, 8);
      Mat roi = perspectiveTransform(image, rect);
      // for each region of interest, check if it met the feature of a position box in QR code
      if (xAxisFeature(roi) && yAxisFeature(roi)) {
        positionBoxCnt++;
        printf("%dth contour is a position box!\n", t);
//        imwrite("../images/qr_roi" + to_string(t) + ".png", roi);
        drawContours(image, contours, static_cast<int>(t), Scalar(255, 0, 0), 2, 8);
        drawContours(binaryPositionBoxesMat, contours, static_cast<int>(t), Scalar(255), 2, 8);
      }
    }
  }

  vector<Point> pts;
  // if there are less than 3 position boxes recognized, simply return
  if (positionBoxCnt != 3) {
    cout << "Position Box Found: " << positionBoxCnt << ". Detection Failed!" << endl;
  } else {
    // scan and save all key points of the three boxes into a vector and return
    for (int row = 0; row < binaryPositionBoxesMat.rows; row++) {
      for (int col = 0; col < binaryPositionBoxesMat.cols; col++) {
        int pv = binaryPositionBoxesMat.at<uchar>(row, col);
        if (pv == 255) {
          pts.push_back(Point(col, row));
        }
      }
    }
  }
  return pts;
}

bool xAxisFeature(Mat &qr_roi) {
  Mat gray, binary;
  cvtColor(qr_roi, gray, COLOR_BGR2GRAY);
  threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);

  int innerBoxSideLength;
  int leftWhiteWidth = 0, rightWhiteWidth = 0;
  int leftBlackWidth = 0, rightBlackWidth = 0;

  int width = binary.cols;
  int height = binary.rows;
  int centerY = height / 2;
  int centerX = width / 2;

  // if the center pixel is not black in the binary image, simply return false since it is invalid
  int pixelValue = binary.at<uchar>(centerY, centerX);
  if (pixelValue == 255) return false;

  // Check if the region of interest has the feature
  bool findLeft = false, findRight = false;
  int start = 0, end = 0;
  int offset = 0;
  while (true) {
    offset++;
    if ((centerX - offset) <= width / 8 || (centerX + offset) >= width - 1) {
      start = -1;
      end = -1;
      break;
    }
    // go left until find a white pixel
    pixelValue = binary.at<uchar>(centerY, centerX - offset);
    if (pixelValue == 255) {
      start = centerX - offset;
      findLeft = true;
    }
    // go right until find a white pixel
    pixelValue = binary.at<uchar>(centerY, centerX + offset);
    if (pixelValue == 255) {
      end = centerX + offset;
      findRight = true;
    }
    // if reaching the white pixel in both the left part and right part, jump out of the loop
    if (findLeft && findRight) {
      break;
    }
  }

  if (start <= 0 || end <= 0) {
    return false;
  }
  innerBoxSideLength = end - start;

  // go left from the "start" point until meeting the black pixel again, which gives the width of left white area
  for (int col = start; col > 0; col--) {
    pixelValue = binary.at<uchar>(centerY, col);
    if (pixelValue == 0) {
      leftWhiteWidth = start - col;
      break;
    }
  }
  // go right from the "end" point until meeting the black pixel again, which gives the width of right white area
  for (int col = end; col < width - 1; col++) {
    pixelValue = binary.at<uchar>(centerY, col);
    if (pixelValue == 0) {
      rightWhiteWidth = col - end;
      break;
    }
  }
  // go right from the right side "white-black" edge until meeting the white pixel(outside the border) again or touch the border, which gives the width of right black area
  for (int col = (end + rightWhiteWidth); col < width; col++) {
    pixelValue = binary.at<uchar>(centerY, col);
    if (pixelValue == 255) {
      rightBlackWidth = col - end - rightWhiteWidth;
      break;
    } else {
      rightBlackWidth++;
    }
  }
  // go left from the left side "white-black" edge until meeting the white pixel(outside the border) again or touch the border, which gives the width of left black area
  for (int col = (start - leftWhiteWidth); col > 0; col--) {
    pixelValue = binary.at<uchar>(centerY, col);
    if (pixelValue == 255) {
      leftBlackWidth = start - col - leftWhiteWidth;
      break;
    } else {
      leftBlackWidth++;
    }
  }

  // For each area's width, do the normalization and multiply by 7(since we assume the feature is 1:1:3:1:1), add by 0.5 for higher fault tolerance
  float sum = innerBoxSideLength + leftBlackWidth + rightBlackWidth + leftWhiteWidth + rightWhiteWidth;
  innerBoxSideLength = static_cast<int>((innerBoxSideLength / sum) * 7.0 + 0.5);
  leftBlackWidth = static_cast<int>((leftBlackWidth / sum) * 7.0 + 0.5);
  rightBlackWidth = static_cast<int>((rightBlackWidth / sum) * 7.0 + 0.5);
  leftWhiteWidth = static_cast<int>((leftWhiteWidth / sum) * 7.0 + 0.5);
  rightWhiteWidth = static_cast<int>((rightWhiteWidth / sum) * 7.0 + 0.5);
  printf(
      "innerBoxSideLength : %d, leftBlackWidth = %d, rightBlackWidth = %d, leftWhiteWidth = %d, rightWhiteWidth = %d\n",
      innerBoxSideLength,
      leftBlackWidth,
      rightBlackWidth,
      leftWhiteWidth,
      rightWhiteWidth);

  if ((innerBoxSideLength == 3 || innerBoxSideLength == 4) && leftBlackWidth == rightBlackWidth
      && leftWhiteWidth == rightWhiteWidth && leftWhiteWidth == leftBlackWidth && leftBlackWidth == 1) { // 1:1:3:1:1
    imwrite("../results/binary_ROI.png", binary);
    return true;
  } else {
    return false;
  }
}

bool yAxisFeature(Mat &qr_roi) {
  Mat gray, binary;
  cvtColor(qr_roi, gray, COLOR_BGR2GRAY);
  threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
  int width = binary.cols;
  int height = binary.rows;
  int centerY = height / 2;
  int centerX = width / 2;
  int pixelValue;

  // blackCnt should be 1.5 * whiteCnt in theory
  int blackCnt = 0, whiteCnt = 0;
  bool found = true;
  // from the center point to the edge, where the straight path is perpendicular to the edge
  for (int row = centerY; row > 0; row--) {
    pixelValue = binary.at<uchar>(row, centerX);
    // innermost black box half side length
    if (pixelValue == 0 && found) {
      blackCnt++;
    }
      // inner white area width
    else if (pixelValue == 255) {
      found = false;
      whiteCnt++;
    }
  }
  blackCnt = blackCnt * 2;
  return blackCnt * 2 > whiteCnt;
}

Mat perspectiveTransform(Mat &image, RotatedRect &rect) {
  int width = static_cast<int>(rect.size.width);
  int height = static_cast<int>(rect.size.height);
  Mat result = Mat::zeros(height, width, image.type());

  vector<Point> src_corners;
  vector<Point> dst_corners;

  // save 4 vertices of the rectangle into Point vector as source corners, the order is bottomLeft, topLeft, topRight, bottomRight.
  Point2f vertices[4];
  rect.points(vertices);
  for (Point2f v: vertices) {
    src_corners.push_back(v);
  }

  // save the destination corners into a vector by the same order above
  dst_corners.push_back(Point(0, 0));
  dst_corners.push_back(Point(width, 0));
  dst_corners.push_back(Point(width, height));
  dst_corners.push_back(Point(0, height));

  // Finds a perspective transformation between two planes with the default method as RANSAC
  Mat h = findHomography(src_corners, dst_corners);

  // Applies a perspective transformation to the image to get the rectangle after perspective transformation
  warpPerspective(image, result, h, result.size());
  return result;
}
