#include "database.h"

void clear_file(std::string file_name) {
  std::ofstream f(file_name, std::ofstream::out | std::ofstream::trunc);
  if (f.good()) {
    f.open(file_name, std::ofstream::out | std::ofstream::trunc);
  }
  f.close();
}

int open_db(std::fstream &db_file, char mode, std::string file_name) {
  if (db_file.is_open()) {
    db_file.close();
  }
  db_file.open(file_name, mode == 'a' ? std::ios_base::app : std::ios_base::in);
  return 0;
}

bool is_empty(std::fstream &db_file, std::string file_name)
{
  open_db(db_file, 'r', file_name);
  return db_file.peek() == std::fstream::traits_type::eof();
}

int write_features(std::fstream &db_file, const std::string& feature_name, const std::vector<double>& features, std::string file_name) {
  open_db(db_file, 'a', file_name);
  db_file << feature_name << ":";
  for (double value: features) {
    db_file << value << ",";
  }
  db_file << std::endl;
  return 0;
}

int read_features(std::fstream &db_file, std::map<std::string, std::vector<std::vector<double>>> &features, std::string file_name) {
  open_db(db_file, 'r', file_name);
  std::string line;
  while (std::getline(db_file, line)) {
    int colon_index = line.find(':');
    // get the label name by splitting with delimiter as ":"
    std::string label_name = line.substr(0, colon_index);
    if (!features.count(label_name)) {
      std::vector<std::vector<double>> label_features;
      features.insert(std::make_pair(label_name, label_features));
    }
    std::vector<double> single_feature;
    std::string features_str = line.substr(colon_index + 1, line.size());
    int pos;
    while ((pos = features_str.find(',')) != std::string::npos) {
      // get each feature value by splitting with delimiter as ","
      std::string value = features_str.substr(0, pos);
      single_feature.emplace_back(std::atof(value.c_str()));
      features_str.erase(0, pos + 1);
    }
    features[label_name].emplace_back(single_feature);
  }
  return 0;
}

// We will change the value of the vector, must not be the reference
double euclidean_distance(std::vector<double> f1, std::vector<double> f2, std::vector<double> mean, std::vector<double> standard_deviation) {
  double diff_square = 0.;

  for (int i = 0; i < f1.size(); i++) {
    f1[i] = (f1[i] - mean[i]) / standard_deviation[i];
    f2[i] = (f2[i] - mean[i]) / standard_deviation[i];
    diff_square += (f1[i] - f2[i]) * (f1[i] - f2[i]);
  }
  return std::sqrt(diff_square);
}

void mean_stddev_cal(const std::vector<std::vector<double>>& features, std::vector<double> &mean, std::vector<double> &standard_deviation) {
  for (std::vector<double> single_feature: features) {
    for (int i = 0; i < single_feature.size(); i++) {
      mean[i] += single_feature[i];
    }
  }
  for (int i = 0; i < mean.size(); i++) {
    mean[i] /= features.size();
  }

  for (std::vector<double> single_feature: features) {
    for (int i = 0; i < single_feature.size(); i++) {
      standard_deviation[i] += (single_feature[i] - mean[i]) * (single_feature[i] - mean[i]);
    }
  }
  for (int i = 0; i < standard_deviation.size(); i++) {
    standard_deviation[i] /= features.size();
    standard_deviation[i] = std::sqrt(standard_deviation[i]);
  }
}

// read the features and calculate the mean and standard deviation for each feature
void preprocess(std::fstream &db_file, int size, std::map<std::string, std::vector<std::vector<double>>> &features, std::vector<double> &mean, std::vector<double> &standard_deviation, std::string file_name) {
  read_features(db_file, features, file_name);

  // calculate the means for each feature vector
  for (int i = 0; i < size; i++) {
    mean.emplace_back(0.);
    standard_deviation.emplace_back(0.);
  }
  std::vector<std::vector<double>> feature_vectors;
  for (std::pair<std::string, std::vector<std::vector<double>>> p: features) {
    for (const std::vector<double>& single_feature: p.second) {
      feature_vectors.emplace_back(single_feature);
    }
  }
  mean_stddev_cal(feature_vectors, mean, standard_deviation);
}


std::string nearest_neighbor_classifier(std::fstream &db_file, std::vector<double> &target_feature) {
  std::map<std::string, std::vector<std::vector<double>>> features;
  std::vector<double> mean;
  std::vector<double> standard_deviation;
  preprocess(db_file, (int)(target_feature.size()), features, mean, standard_deviation, FEATURE_FILE_NAME);

  double min_dist = 100000.;
  std::string min_label;
  for (std::pair<std::string, std::vector<std::vector<double>>> p: features) {
    for (const std::vector<double>& single_feature: p.second) {
      double distance = euclidean_distance(single_feature, target_feature, mean, standard_deviation);
      if(distance < min_dist) {
        min_dist = distance;
        min_label = p.first;
      }
    }
  }
  return min_label;
}

int knn_classifier(std::fstream &db_file, std::vector<double> &target_feature, int k, std::string &nearest_label) {
  std::map<std::string, std::vector<std::vector<double>>> features;
  std::vector<double> mean;
  std::vector<double> standard_deviation;
  preprocess(db_file, (int)(target_feature.size()), features, mean, standard_deviation, FEATURE_FILE_NAME);

  // for each label, calculate the three nearest neighbor
  std::map<std::string, double> label_distances;
  for (std::pair<std::string, std::vector<std::vector<double>>> same_label_features: features) {
    if (same_label_features.second.size() < k) {
      return -1;
    }
    std::vector<double> distances;
    for (std::vector<double> single_feature: same_label_features.second) {
      double distance = euclidean_distance(single_feature, target_feature, mean, standard_deviation);
      distances.emplace_back(distance);
    }
    double sum_k_distance = 0.;
    for (int i = 0; i < k; i++) {
      sum_k_distance += distances[i];
    }
    label_distances.insert(std::make_pair(same_label_features.first, sum_k_distance));
  }
  double min_dist = 10000.;
//  std::string min_label;
  for (std::pair<std::string, double> p: label_distances) {
    if (p.second < min_dist) {
      nearest_label = p.first;
      min_dist = p.second;
    }
  }
  return 0;
}

int evaluate(std::fstream &db_file, std::fstream &test_file, int k) {
  std::map<std::string, std::vector<std::vector<double>>> train_features;
  read_features(db_file, train_features, FEATURE_FILE_NAME);
  std::map<std::string, std::vector<std::vector<double>>> test_features;
  read_features(test_file, test_features, EVALUATE_FILE_NAME);

  std::map<std::pair<std::string, std::string>, int> evaluation;
  for (std::pair<std::string, std::vector<std::vector<double>>> p: test_features) {
    for (std::pair<std::string, std::vector<std::vector<double>>> q: train_features) {
      evaluation.insert(std::make_pair(std::make_pair(p.first, q.first), 0));
    }
  }

  for (std::pair<std::string, std::vector<std::vector<double>>> single_label_features: test_features) {
    for (std::vector<double> single_test_feature: single_label_features.second) {
      std::string output_label;
      if (k > 1) {
        int status = knn_classifier(db_file, single_test_feature, k, output_label);
        if (status == -1) {
          std::cout << "At least " << k << " examples in each class!" << std::endl;
          return -1;
        }
      } else {
        output_label = nearest_neighbor_classifier(db_file, single_test_feature);
      }
      std::pair<std::string, std::string> key = std::make_pair(single_label_features.first, output_label);
      evaluation[key] += 1;
    }
  }

  // write the evaluation result to a file
  clear_file(EVALUATE_OUTPUT_FILE_NAME);
  open_db(db_file, 'a', EVALUATE_OUTPUT_FILE_NAME);
  db_file << "\t\t";
  std::vector<std::string> header;
  for (std::pair<std::string, std::vector<std::vector<double>>> q: train_features) {
    db_file << q.first << "\t";
    header.emplace_back(q.first);
  }
  db_file << std::endl;

  for (std::pair<std::string, std::vector<std::vector<double>>> p: test_features) {
    db_file << p.first << "\t";
    for (std::string colum: header) {
      db_file << evaluation[std::make_pair(p.first, colum)] << "\t";
    }
    db_file << std::endl;
  }
  db_file.close();
  return 0;
}