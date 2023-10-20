#ifndef SRC_GRAPHNETWORK_H_
#define SRC_GRAPHNETWORK_H_

#include <fstream>

#include "network.h"
#include "neuron.h"

namespace s21 {

class GraphNetwork : public Network {
  typedef enum { kInputLayer, kHiddenLayer, kOutputLayer } layer_type;

 public:
  GraphNetwork();
  explicit GraphNetwork(int num_hidden_layers);
  virtual ~GraphNetwork();

  void Clear();

  bool TrainNetwork(std::ifstream& fp, size_t& count, size_t g_begin,
                    size_t g_end) override;
  bool TestNetwork(std::ifstream& fp, size_t& count, size_t max_tests) override;
  int Predict(const std::vector<int>& input_layer) override;

  std::vector<double>& GetVector() { return vector_; }
  void LoadWeights(const std::string& weights_file) override;
  void SaveWeights(const std::string& weights_file) override;
  size_t GetNumLayers() override { return layers_.size(); }

  void GenerateNetwork(int num_hidden_layers) override;
  void InitNetwork() override;
  void ShowNetwork() override;

 private:
  class Layer {
   public:
    explicit Layer(layer_type t) : type_(t) {
      if (type_ == kInputLayer) {
        neurons_.resize(kHiddenLayerNeurons);
        for (auto& it : neurons_) {
          it.GetWeight().resize(kInputLayerNeurons);
        }
      } else if (type_ == kHiddenLayer) {
        neurons_.resize(kHiddenLayerNeurons);
        for (auto& it : neurons_) {
          it.GetWeight().resize(kHiddenLayerNeurons);
        }
      } else if (type_ == kOutputLayer) {
        neurons_.resize(kOutputLayerNeurons);
        for (auto& it : neurons_) {
          it.GetWeight().resize(kHiddenLayerNeurons);
        }
      } else {
        throw std::invalid_argument("Error: unknown type of layers");
      }
    }
    ~Layer() {}
    layer_type GetType() { return type_; }
    std::vector<s21::Neuron>& GetNeurons() { return neurons_; }

    void Load(std::ifstream* fp);
    void Save(std::ofstream* fp);

   private:
    layer_type type_;
    std::vector<s21::Neuron> neurons_{};
  };

  std::vector<Layer*> layers_;
  std::vector<double> vector_{};

  double Sigmoid_(double value) { return (1.0 / (1.0 + exp(-value))); }
  int MaxElement_();
  void EmnistLetterToVector_();
  void CalculateVector_();
  void CalculateDeltaWeights_(size_t expected);
  void UpdateWeights_();
};

}  // namespace s21

#endif  // SRC_GRAPHNETWORK_H_
