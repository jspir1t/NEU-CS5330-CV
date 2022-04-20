#include "utils.h"

using namespace cv;
using namespace std;

int main() {
  printf("OpenCV: %s", cv::getBuildInformation().c_str());
}