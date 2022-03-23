#include <opencv2/opencv.hpp>

#ifndef PROJ3_INCLUDE_SEGEMENTATION_H_
#define PROJ3_INCLUDE_SEGEMENTATION_H_

#define BACK_GROUND 0
#define FRONT_GROUND 255
#define THRESHOLD 100

/**
 * If the pixel value is bigger than threshold, it will belong to background, else front ground
 * @param src the original mat
 * @param dst the thresholding mat
 * @param threshold the threshold for all the three rgb values
 * @return 0 if success
 */
int threshold(const cv::Mat &src, cv::Mat &dst, int threshold);

/**
 * Clean up the thresholding image with shrink first, then grow the same steps
 * @param src the thresholding mat
 * @param dst the cleanup mat
 * @param steps the steps to be shrink or grow
 * @return 0 if success
 */
int cleanup(cv::Mat &src, cv::Mat &dst, int steps);

/**
 * Segment the cleaned up mat to get @param component_num components, it will store the seperated components in @param
 * regions and the largest component in @param major(excluding the background component). If the area of certain
 * component is less than the @param min_area, it will be discard and not shown in the window. Moreover, I have already
 * discard the components that are adjacent to the mat border.
 * @param src the cleaned up mat
 * @param dst the mat contains @param component_num components and colored with different colors
 * @param component_num the maximum number of component
 * @param regions the map stores component label with its component mat
 * @param min_area the minimum of the required component
 * @param major the component with the largest area size in the @param regions
 * @return -1 if there is only background component or no component after removing the border adjacent component and
 * small components that are less than @param min_area
 */
int segment(cv::Mat &src,
            cv::Mat &dst,
            int component_num,
            std::map<int, cv::Mat> &regions,
            int min_area,
            cv::Mat &major);

/**
 * Mark the component with bounding box and the axis passing through the centroid with all green lines and circle
 * @param segmented_img the segmented mat with different colors on different components
 * @param draw_vertices the seven vertices representing the corners of the bounding box, the centroid and the two
 * vertices for the central axis
 */
void mark_object(cv::Mat &segmented_img, std::vector<cv::Point> draw_vertices);

/**
 * Calculate the features for a given component mat and store them in the order of (height/width, fill_ration,
 * u_22_alpha, hu_0, hu_1, hu_2). Generate the required vertices and store them in the @param draw_vertices
 * @param src the mat with only one component in it
 * @param feature_vector the feature vector
 * @param draw_vertices the vertices for mark a component
 * @return 0 if success
 */
int features(cv::Mat &src, std::vector<double> &feature_vector, std::vector<cv::Point> &draw_vertices);

#endif //PROJ3_INCLUDE_SEGEMENTATION_H_
