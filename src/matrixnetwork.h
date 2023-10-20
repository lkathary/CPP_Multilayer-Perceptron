#ifndef SRC_MATRIXNETWORK_H_
#define SRC_MATRIXNETWORK_H_

#include <fstream>
#include <vector>

#include "matrix.h"
#include "network.h"

namespace s21 {

class MatrixNetwork : public Network {
  typedef enum { kInputLayer, kHiddenLayer, kOutputLayer } layer_type;

 public:
  MatrixNetwork();
  explicit MatrixNetwork(int num_hidden_layers);
  virtual ~MatrixNetwork();

  void Clear();

  bool TrainNetwork(std::ifstream& fp, size_t& count, size_t g_begin,
                    size_t g_end) override;
  bool TestNetwork(std::ifstream& fp, size_t& count, size_t max_tests) override;
  int Predict(const std::vector<int>& input_layer) override;
  // std::pair<int, double> Predict(const std::vector<int>& input_layer);

  void LoadWeights(const std::string& weights_file) override;
  void SaveWeights(const std::string& weights_file) override;
  size_t GetNumLayers() override { return layers_.size(); }

  void GenerateNetwork(int num_hidden_layers) override;
  void InitNetwork() override;
  void ShowNetwork() override;

 private:
  class Layer {
   public:
    explicit Layer(layer_type t) : type_(t), vector_(new Matrix) {
      if (type_ == kInputLayer) {
        weights_ = new Matrix(kInputLayerNeurons, kHiddenLayerNeurons);
        delta_weights_ = new Matrix(1, kHiddenLayerNeurons);
      } else if (type_ == kHiddenLayer) {
        weights_ = new Matrix(kHiddenLayerNeurons, kHiddenLayerNeurons);
        delta_weights_ = new Matrix(1, kHiddenLayerNeurons);
      } else if (type_ == kOutputLayer) {
        weights_ = new Matrix(kHiddenLayerNeurons, kOutputLayerNeurons);
        delta_weights_ = new Matrix(1, kOutputLayerNeurons);
      } else {
        weights_ = nullptr;
        delta_weights_ = nullptr;
      }
    }
    ~Layer() {
      delete weights_;
      delete vector_;
      delete delta_weights_;
    }
    layer_type GetType() { return type_; }
    Matrix* GetMatrix() { return weights_; }
    Matrix* GetVector() { return vector_; }
    Matrix* GetDelta() { return delta_weights_; }

   private:
    layer_type type_;
    Matrix* weights_;
    Matrix* vector_;
    Matrix* delta_weights_;
  };

  std::vector<Layer*> layers_;

  Matrix* EmnistLetterToVector_();
  void CalculateVector_(Matrix* vector);
  void CalculateDeltaWeights_(int expected);
  void UpdateWeights_();
};

}  // namespace s21

#endif  //   SRC_MATRIXNETWORK_H_
