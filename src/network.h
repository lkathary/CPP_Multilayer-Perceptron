#ifndef SRC_NETWORK_H_
#define SRC_NETWORK_H_

#include <vector>

#include "matrix.h"

namespace s21 {

const size_t kNumDataSetSamples = 88800;
const size_t kDataSetBatchSize = 1000;
const size_t kNumDataSetTests = 14800;

const std::string kDataSetTrain = "./datasets/emnist-letters-train.csv";
const std::string kDataSetTrain_ = "./datasets/emnist-letters-myself2.csv";

const std::string kDataSetTest = "./datasets/emnist-letters-test.csv";
const std::string kWeightsFile = "./weights/weights_2_784.txt";

const int kSizeImage = 512;

const int kMinHiddenLayers = 2;
const int kMaxHiddenLayers = 5;

const int kNumNeurons = 28;

const int kInputLayerNeurons = 784;
const int kOutputLayerNeurons = 26;
const int kHiddenLayerNeurons = 100;
const int kNumHiddenLayers = 2;

typedef enum { kMatrixNet, kGraphNet } net_type;

class Network {
 public:
  Network()
      : type_(kMatrixNet),
        learning_rate_(0.4),
        count_errors_(0),
        confusion_matrix_(
            new Matrix(kOutputLayerNeurons, kOutputLayerNeurons)) {}
  ~Network() { delete confusion_matrix_; }

  net_type GetType() { return type_; }

  void virtual LoadWeights(const std::string& weights_file) = 0;
  void virtual SaveWeights(const std::string& weights_file) = 0;

  size_t virtual GetNumLayers() = 0;
  int GetInputLayerNeurons() { return kInputLayerNeurons; }
  int GetHiddenLayerNeurons() { return kHiddenLayerNeurons; }
  int GetOutputLayerNeurons() { return kOutputLayerNeurons; }

  void virtual GenerateNetwork(int num_hidden_layers) = 0;
  void virtual InitNetwork() = 0;
  void virtual ShowNetwork() = 0;

  std::vector<int>& GetEmnistLetter() { return emnist_letter_; }
  void ReadEmnistLetter(const std::string& line);

  void SetLearningRate(double lr) { learning_rate_ = lr; }

  bool virtual TrainNetwork(std::ifstream& fp, size_t& count, size_t g_begin,
                            size_t g_end) = 0;
  bool virtual TestNetwork(std::ifstream& fp, size_t& count,
                           size_t max_tests) = 0;
  int virtual Predict(const std::vector<int>& input_layer) = 0;

  //  Statistics
  //  https://towardsdatascience.com/precision-recall-and-f1-score-of-multiclass-classification-learn-in-depth-6c194b217629

  double CalculateAccuracy();
  double CalculatePrecision();
  double CalculateRecall();
  double CalculateFmeasure();

  void ResetStatistics();
  size_t GetCountErrors() { return count_errors_; }
  void ShowConfusionMatrix() { confusion_matrix_->Show(); }

 protected:
  net_type type_;
  std::vector<int> emnist_letter_;
  double learning_rate_;
  size_t count_errors_;
  s21::Matrix* confusion_matrix_;
};

}  // namespace s21

#endif  //   SRC_NETWORK_H_
