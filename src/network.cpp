#include "network.h"

namespace s21 {

void Network::ReadEmnistLetter(const std::string& line) {
  emnist_letter_.clear();
  std::string substr;
  try {
    for (auto it : line) {
      if (it != ',') {
        substr.push_back(it);
      } else {
        emnist_letter_.push_back(std::stoi(substr));
        substr.clear();
        ++it;
      }
    }
    emnist_letter_.push_back(std::stoi(substr));
  } catch (const std::exception& e) {
    throw std::invalid_argument("Error, incorrect dataset format");
  }
  if (emnist_letter_.size() != kInputLayerNeurons + 1) {
    throw std::length_error("Error, incorrect dataset format");
  }
}

//  Statistics
//  https://towardsdatascience.com/precision-recall-and-f1-score-of-multiclass-classification-learn-in-depth-6c194b217629

double Network::CalculateAccuracy() {
  int total = 0, true_positives = 0;
  for (int i = 0; i < kOutputLayerNeurons; ++i) {
    for (int j = 0; j < kOutputLayerNeurons; ++j) {
      total += (*confusion_matrix_)(i, j);
      if (i == j) {
        true_positives += (*confusion_matrix_)(i, j);
      }
    }
  }
  return static_cast<double>(true_positives) / total;
}

double Network::CalculatePrecision() {
  int total[kOutputLayerNeurons]{}, true_positives[kOutputLayerNeurons]{};
  for (int i = 0; i < kOutputLayerNeurons; ++i) {
    for (int j = 0; j < kOutputLayerNeurons; ++j) {
      total[j] += (*confusion_matrix_)(i, j);
      if (i == j) {
        true_positives[j] = (*confusion_matrix_)(i, j);
      }
    }
  }
  double precision = 0;
  int exist_test = 0;
  for (int i = 0; i < kOutputLayerNeurons; ++i) {
    if (total[i] > 0) {
      precision += static_cast<double>(true_positives[i]) / total[i];
      ++exist_test;
    }
  }
  return precision / exist_test;
}

double Network::CalculateRecall() {
  int total[kOutputLayerNeurons]{}, true_positives[kOutputLayerNeurons]{};
  for (int i = 0; i < kOutputLayerNeurons; ++i) {
    for (int j = 0; j < kOutputLayerNeurons; ++j) {
      total[i] += (*confusion_matrix_)(i, j);
      if (i == j) {
        true_positives[i] = (*confusion_matrix_)(i, j);
      }
    }
  }
  double precision = 0;
  int exist_test = 0;
  for (int i = 0; i < kOutputLayerNeurons; ++i) {
    if (total[i] > 0) {
      precision += static_cast<double>(true_positives[i]) / total[i];
      ++exist_test;
    }
  }
  return precision / exist_test;
}

double Network::CalculateFmeasure() {
  return CalculatePrecision() * CalculateRecall() * 2 /
         (CalculatePrecision() + CalculateRecall());
}

void Network::ResetStatistics() {
  count_errors_ = 0;
  for (int i = 0; i < kOutputLayerNeurons; ++i) {
    for (int j = 0; j < kOutputLayerNeurons; ++j) {
      (*confusion_matrix_)(i, j) = 0;
    }
  }
}

}  // namespace s21
