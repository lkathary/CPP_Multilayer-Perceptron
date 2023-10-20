#ifndef SRC_CONTROLLER_H_
#define SRC_CONTROLLER_H_

#include "graphnetwork.h"
#include "matrixnetwork.h"

namespace s21 {

class Controller {
 public:
  static Controller* GetInstance() {
    if (!controller_) {
      controller_ = new Controller();
    }
    return controller_;
  }
  void Connect(s21::MatrixNetwork* mn, s21::GraphNetwork* gn) {
    matrix_instance_ = mn;
    graph_instance_ = gn;
    current_network_ = matrix_instance_;
  }
  ~Controller() {}
  void SetCurrentNetwork(net_type t) {
    if (t == s21::kMatrixNet) {
      current_network_ = matrix_instance_;
    } else {
      current_network_ = graph_instance_;
    }
  }

  void GenerateNetwork(int num_hidden_layers) {
    current_network_->GenerateNetwork(num_hidden_layers);
    current_network_->InitNetwork();
  }

  void ShowNetwork() { current_network_->ShowNetwork(); }
  void ShowConfusionMatrix() { current_network_->ShowConfusionMatrix(); }

  s21::net_type GetType() { return current_network_->GetType(); }
  size_t GetNumLayers() { return current_network_->GetNumLayers(); }
  int GetInputLayerNeurons() {
    return current_network_->GetInputLayerNeurons();
  }
  int GetHiddenLayerNeurons() {
    return current_network_->GetHiddenLayerNeurons();
  }
  int GetOutputLayerNeurons() {
    return current_network_->GetOutputLayerNeurons();
  }

  std::vector<int>& GetEmnistLetter() {
    return current_network_->GetEmnistLetter();
  }

  void SetLearningRate(double lr) { current_network_->SetLearningRate(lr); }

  size_t GetCountErrors() { return current_network_->GetCountErrors(); }
  double CalculateAccuracy() {
    return current_network_->CalculateAccuracy() * 100;
  }
  double CalculatePrecision() {
    return current_network_->CalculatePrecision() * 100;
  }
  double CalculateRecall() { return current_network_->CalculateRecall() * 100; }
  double CalculateFmeasure() {
    return current_network_->CalculateFmeasure() * 100;
  }
  void ResetStatistics() { current_network_->ResetStatistics(); }

  void SaveWeights(const std::string& weights_file) {
    current_network_->SaveWeights(weights_file);
  }
  std::string LoadWeights(const std::string& weights_file) {
    try {
      current_network_->LoadWeights(weights_file);
      return "Weights uploaded successfully";
    } catch (const std::exception& e) {
      return e.what();
    }
  }

  bool TrainNetwork(std::ifstream& fp, size_t& count, size_t g_begin,
                    size_t g_end) {
    return current_network_->TrainNetwork(fp, count, g_begin, g_end);
  }

  bool TestNetwork(std::ifstream& fp, size_t& count, size_t max_tests) {
    return current_network_->TestNetwork(fp, count, max_tests);
  }

  int Predict(const std::vector<int>& input_layer) {
    return current_network_->Predict(input_layer);
  }

 private:
  static Controller* controller_;
  s21::MatrixNetwork* matrix_instance_;
  s21::GraphNetwork* graph_instance_;
  s21::Network* current_network_;

  Controller() {}
};

}  //   namespace s21

#endif  //   SRC_CONTROLLER_H_
