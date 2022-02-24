#include "database.h"

void clear_file() {
  std::ofstream f(FEATURE_FILE_NAME, std::ofstream::out | std::ofstream::trunc);
  if (f.good()) {
    f.open(FEATURE_FILE_NAME, std::ofstream::out | std::ofstream::trunc);
  }
  f.close();
}

int open_db(std::fstream &db_file, char mode) {
  if (db_file.is_open()) {
    db_file.close();
  }
  db_file.open(FEATURE_FILE_NAME, mode == 'a' ? std::ios_base::app : std::ios_base::in);
  return 0;
}

bool is_empty(std::fstream &db_file)
{
  open_db(db_file, 'r');
  return db_file.peek() == std::fstream::traits_type::eof();
}

int write_features(std::fstream &db_file, const std::string& feature_file_name, const std::vector<double>& features) {
  open_db(db_file, 'a');
  db_file << feature_file_name << ":";
  for (double value: features) {
    db_file << value << ",";
  }
  db_file << std::endl;
  return 0;
}

int read_features(std::fstream &db_file, std::map<std::string, std::vector<double>> &features) {
  open_db(db_file, 'r');
  std::string line;
  while (std::getline(db_file, line)) {
    int colon_index = line.find(':');
    // get the label name by splitting with delimiter as ":"
    std::string label_name = line.substr(0, colon_index);
    std::vector<double> label_features;
    features.insert(std::make_pair(label_name, label_features));
    std::string features_str = line.substr(colon_index + 1, line.size());
    int pos;
    while ((pos = features_str.find(',')) != std::string::npos) {
      // get each feature value by splitting with delimiter as ","
      std::string value = features_str.substr(0, pos);
      features[label_name].emplace_back(std::atof(value.c_str()));
      features_str.erase(0, pos + 1);
    }
  }
  return 0;
}

// We will change the value of the vector, must not be the reference
double euclidean_distance(std::vector<double> f1, double mean1, double std_dev1, std::vector<double> f2, double mean2, double std_dev2) {
  double diff_square = 0.;

//  std::cout << "f1: ";
//  for (auto i: f1) {
//    std::cout << i << ", ";
//  }
//  std::cout << std::endl;
//
//  std::cout << "f2: ";
//  for (auto i: f2) {
//    std::cout << i << ", ";
//  }
//  std::cout << std::endl;


  for (int i = 0; i < f1.size(); i++) {
    f1[i] = (f1[i] - mean1) / std_dev1;
    f2[i] = (f2[i] - mean2) / std_dev2;
    diff_square += (f1[i] - f2[i]) * (f1[i] - f2[i]);
  }
  std::cout << "result: " << std::sqrt(diff_square) << std::endl;
  return std::sqrt(diff_square);
}

std::string euclidean_classifier(std::fstream &db_file, std::vector<double> &target_feature) {
  std::map<std::string, std::vector<double>> features;
  read_features(db_file, features);

  // calculate the means for each feature vector
  std::map<std::string, double> means;
  std::map<std::string, double> standard_deviations;
  for (std::pair<std::string, std::vector<double>> p: features) {
    double sum = 0.;
    for (double value: p.second) {
      sum += value;
    }
    means[p.first] = sum / (double)(p.second.size());

    double variance = 0.;
    for (double value: p.second) {
      variance += (value - means[p.first]) * (value - means[p.first]);
    }
    variance /= (double)(p.second.size());
    standard_deviations[p.first] = std::sqrt(variance);
  }

  // calculate mean and standard deviation for target feature
  double sum = 0.;
  for (double value: target_feature) {
    sum += value;
  }
  double mean = sum / (double)(target_feature.size());
  double variance = 0.;
  for (double value: target_feature) {
    variance += (value - mean) * (value - mean);
  }
  variance /= (double)(target_feature.size());
  double standard_deviation = std::sqrt(variance);

  double min_dist = 100000.;
  std::string min_label;
  for (std::pair<std::string, std::vector<double>> p: features) {
    double distance = euclidean_distance(p.second, means[p.first], standard_deviations[p.first], target_feature, mean, standard_deviation);
    if(distance < min_dist) {
      min_dist = distance;
      min_label = p.first;
    }
  }


//  for(auto i: features) {
//    std::cout << i.first << ": ";
//    for (auto j: i.second) {
//      std::cout << j << ",";
//    }
//  }
//  std::cout << std::endl;
  return min_label;
}