#include <fstream>
#include <vector>
#include "iostream"
#include <map>
#include "math.h"
#include "algorithm"
#include <queue>

#ifndef PROJ3_INCLUDE_DATABASE_H_
#define PROJ3_INCLUDE_DATABASE_H_

#define FEATURE_FILE_NAME "../features.txt"
#define EVALUATE_FILE_NAME "../test_features.txt"
#define EVALUATE_OUTPUT_FILE_NAME "../evaluation.txt"

/**
 * Check if the @param file_name is empty
 * @param db_file the fstream
 * @param file_name  the name of the file to be checked
 * @return true if it is empty else false
 */
bool is_empty(std::fstream &db_file, std::string file_name);

/**
 * clear the content in the @param file_name
 * @param file_name the name of the file to be cleared
 */
void clearFile(std::string file_name);
/**
 * Write features to the @param file_name, each row should be a @param feature_name along with a @param features
 * @param db_file the fstream
 * @param feature_name the name of the feature
 * @param features the vector storing the feature values
 * @param file_name the name of the file
 * @return 0 if success
 */
int write_features(std::fstream &db_file,
                   const std::string &feature_name,
                   const std::vector<double> &features,
                   std::string file_name);

/**
 * The nearest neighbor classifier to get the nearest feature for @param target_feature in the feature file
 * @param db_file the file storing the features of the training data
 * @param target_feature the target feature
 * @return the corresponding label for the nearest feature
 */
std::string nearest_neighbor_classifier(std::fstream &db_file, std::vector<double> &target_feature);

/**
 * The KNN classifier to get the nearest feature for @param target_feature in the feature file
 * @param db_file the file storing the features of the training data
 * @param target_feature  * @param target_feature
 * @param k the number of nearest neighbors for each class need to be checked
 * @param nearest_label the string representing the nearest feature label
 * @return 0 if success
 */
int knn_classifier(std::fstream &db_file, std::vector<double> &target_feature, int k, std::string &nearest_label);

/**
 * Evaluate the training set in the @param test_file with the training features in the @param db_file
 * @param db_file the fstream for operating the training dataset
 * @param test_file the fstream for operating the test dataset
 * @param k the number of nearest neighbors need to be checked, if it is bigger than 2, reuse the knn classifier above,
 * else reuse the nearest neighbor classifier
 * @return 0 if success
 */
int evaluate(std::fstream &db_file, std::fstream &test_file, int k);

#endif //PROJ3_INCLUDE_DATABASE_H_
