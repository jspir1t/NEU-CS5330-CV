/**
 * Main CPP file, contains a main function in which it will read the command line parameters to determine the mode. If
 * it is video mode, display a video window and draw the bounding box once a QR code is detected. If it is single image
 * mode, display a static window to show the result. If it is evaluation mode, it will go over the whole dataset to
 * detect the QR code and write the results into results folder.
 */
#include "utils.h"
#include <dirent.h>
#include <fstream>
#include <opencv2/core/utils/filesystem.hpp>

using namespace cv;
using namespace std;

string mode;
string method = "self";

/**
 * Real time QR code detection and segmentation in Video
 * @return -1 if the video device is not found, else 0
 */
int videoDetectSegment() {
  cv::VideoCapture *capdev;
  // open the video device
  capdev = new cv::VideoCapture(0);
  if (!capdev->isOpened()) {
    printf("Unable to open video device\n");
    return (-1);
  }
  // get some properties of the image
  cv::Size refS((int) capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
  printf("Expected size: %d %d\n", refS.width, refS.height);
  cv::Mat frame;
  std::vector<double> single_feature_vector;
  for (;;) {
    *capdev >> frame; // get a new frame from the camera, treat as a stream
    if (frame.empty()) {
      printf("frame is empty\n");
      break;
    }
    vector<Point> pts = detectAndDecode(frame, true);
    // draw the bbox for the detected rectangles
    if (!pts.empty()) {
      drawBBox(frame, pts);
    }
    // see if there is a waiting keystroke
    int key = cv::waitKey(10);
    // if user types 'q', quit the program
    if (key == 'q') {
      break;
    }
    if (key == 's') {
      String file_name = "../results/" + mode + "_" + method + ".png";
      imwrite(file_name, frame);
      cout << "This image is saved as " << file_name << endl;
    }

    cv::imshow("frame", frame);
  }
  return 0;
}

/**
 * Evaluate the results with the whole images directory as input. The result will be written to results/self.csv, output
 * images will be saved at results/self_eval
 */
void imagesEvaluation() {
  String datasetDir = "../images/";
  String dstDir = "../results/self_eval/";
  cv::utils::fs::createDirectory(dstDir);
  // if directory exists or created successfully
  if (auto dir = opendir(datasetDir.c_str())) {
    map<String, bool> result;
    // iterate th files and directories
    while (auto f = readdir(dir)) {
      // Skip everything that starts with a dot
      if (!f->d_name || f->d_name[0] == '.')
        continue;

      String fileName = string(f->d_name);
      String fileDir = datasetDir + fileName;
      Mat image = cv::imread(fileDir);
      vector<Point> pts = detectAndDecode(image, false);
      if (!pts.empty()) {
        result.insert(pair<String, bool>(string(f->d_name), true));
        drawBBox(image, pts);
      } else {
        result.insert(pair<String, bool>(string(f->d_name), false));
      }
      cv::imwrite(dstDir + mode + "_" + method + "_" + fileName, image);
    }
    for (const auto &element: result) {
      cout << element.first << ": " << (element.second ? "True" : "False") << ",\t";
    }
    writeToCSV(result);
    closedir(dir);
  }
}

/**
 * Given a image name, detect and segment the QR code in that image
 * @param fileName the image name without .png
 */
void imageDetectSegment(String fileName) {
  String datasetDir = "../images/";
  String fileDir = datasetDir + fileName;
  ifstream f(fileDir);
  // check if the image exists. Exit with -1 code if not found
  if (!f.good()) {
    cout << "Error: File not exist!" << endl;
    exit(-1);
  }

  Mat img = cv::imread(fileDir);
  vector<Point> pts = detectAndDecode(img, true);
  if (!pts.empty()) {
    drawBBox(img, pts);
  }
  // display the result in a window
  while (true) {
    cv::imshow("image", img);
    // see if there is a waiting keystroke
    int key = cv::waitKey(10);
    // if user types 'q', quit the program
    if (key == 'q') {
      break;
    }
    if (key == 's') {
      String file_name = "../results/" + mode + "_" + method + "_" + fileName;
      imwrite(file_name, img);
      cout << "This image is saved as " << file_name << endl;
    }
  }

}

/**
 * Parse the argument to determine the mode, there are three modes: video, image and eval.
 * @param argc the number of arguments
 * @param argv the argument array
 * @return 0 if works fine, else -1
 */
int main(int argc, char *argv[]) {
  if (argc < 2 || argc >= 4) {
    std::cout << "Usage: ./main.exe video/eval || ./main.exe image <image_name>" << std::endl;
    exit(-1);
  }
  String m = argv[1];
  if (m == "video") {
    mode = "video";
    videoDetectSegment();
  } else if (m == "eval") {
    mode = "eval";
    imagesEvaluation();
  } else if (m == "image" && argc == 3) {
    mode = "image";
    imageDetectSegment(string(argv[2]) + ".png");
  } else {
    cout << "Invalid parameters!" << endl;
    exit(-1);
  }
  return 0;
}