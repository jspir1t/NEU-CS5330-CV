#include <fstream>
#include <vector>
#include "iostream"
#include <map>
#include "math.h"

#ifndef PROJ3_INCLUDE_DATABASE_H_
#define PROJ3_INCLUDE_DATABASE_H_

#define FEATURE_FILE_NAME "../features.txt"

bool is_empty(std::fstream &db_file);
void clear_file();
int write_features(std::fstream &db_file, const std::string& feature_file_name, const std::vector<double>& features);
std::string euclidean_classifier(std::fstream &db_file, std::vector<double> &target_feature);

#endif //PROJ3_INCLUDE_DATABASE_H_
