/**
 * Header file of the utils file, define the function signatures.
 */
#include <opencv2/opencv.hpp>
#include <iostream>

#ifndef FINAL_PROJECT_INCLUDE_UTILS_H_
#define FINAL_PROJECT_INCLUDE_UTILS_H_
using namespace cv;
using namespace std;

/**
 * Write the evaluation result into a csv file.
 * @param map the evaluation result where key is the file name and value is 1 for successfully detected, 0 for failure
 */
void writeToCSV(const map<String, bool>& map);

/**
 * Given an image, detect the qr code and return a vector of points of the three position boxes. If not detected, return
 * an empty vector.
 * @param image the original image
 * @param showWin show the binary image window if true
 * @return a vector of points of the three position boxes
 */
vector<Point> detectAndDecode(Mat &image, bool showWin);

/**
 * Draw the bing bounding boxes onto the image based on the position boxes vertices
 * @param image the original image
 * @param pts a vector of points of the three position boxes
 */
void drawBBox(Mat &image, vector<Point> &pts);

/**
 * Check if a region of interest meet the requirement that the width ratio of those black and white areas is 1:1:3:1:1
 * along the center X axis
 * @param qr_roi the region of interest extracted from the original image and after perspective projection
 * @return true if it meets the requirement above else false
 */
bool xAxisFeature(Mat &qr_roi);

/**
 * Check if a region of interest meet the requirement that the width of inner black box is smaller than the width of
 * white area along the center Y axis
 * @param qr_roi the region of interest extracted from the original image and after perspective projection
 * @return true if it meets the requirement above else false
 */
bool yAxisFeature(Mat &qr_roi);

/**
 * Given an image and a rotated rectangle, return the rectangle after perspective transformation
 * @param image the whole image
 * @param rect the rotated rectangle inside the image
 * @return the same size of mat as @param rect after perspective transformation
 */
Mat perspectiveTransform(Mat &image, RotatedRect &rect);
#endif //FINAL_PROJECT_INCLUDE_UTILS_H_
