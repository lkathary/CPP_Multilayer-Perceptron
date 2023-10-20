#ifndef SRC_NEURON_H_
#define SRC_NEURON_H_

#include <iostream>
#include <vector>

namespace s21 {

class Neuron {
 public:
  Neuron() : value_(0), delta_(0) {}

  double& GetValue() { return value_; }
  double& GetDelta() { return delta_; }
  std::vector<double>& GetWeight() { return weight_; }
  std::vector<Neuron*>& GetInput() { return input_; }

  void ShowInputNeurons() {
    for (auto& it : input_) {
      std::cout << "[" << it << "]";
    }
  }

 private:
  std::vector<Neuron*> input_{};
  std::vector<double> weight_{};
  double value_;
  double delta_;
};

}  // namespace s21

#endif  //  SRC_NEURON_H_
