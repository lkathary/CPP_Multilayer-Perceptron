#include "graphnetwork.h"

namespace s21 {

GraphNetwork::GraphNetwork() : GraphNetwork(kNumHiddenLayers) {}

GraphNetwork::GraphNetwork(int num_hidden_layers) {
  type_ = kGraphNet;
  srand(time(0));
  GenerateNetwork(num_hidden_layers);
}

GraphNetwork::~GraphNetwork() {
  for (auto& it : layers_) {
    delete it;
  }
}

void GraphNetwork::Clear() {
  for (auto& it : layers_) {
    delete it;
  }
  layers_.clear();
}

void GraphNetwork::GenerateNetwork(int num_hidden_layers) {
  Clear();
  layers_.push_back(new Layer(kInputLayer));
  for (int i = 0; i < num_hidden_layers; ++i) {
    layers_.push_back(new Layer(kHiddenLayer));
  }
  layers_.push_back(new Layer(kOutputLayer));

  for (auto rit = layers_.rbegin(); rit != layers_.rend(); ++rit) {
    if ((*rit)->GetType() != kInputLayer) {
      auto rit_prev = rit;
      ++rit_prev;
      for (auto& it_n : (*rit)->GetNeurons()) {
        it_n.GetInput().resize((*rit_prev)->GetNeurons().size());
        for (size_t i = 0; i < it_n.GetInput().size(); ++i) {
          for (size_t j = 0; j < (*rit_prev)->GetNeurons().size(); ++j) {
            if (i == j) {
              it_n.GetInput()[i] = &((*rit_prev)->GetNeurons()[j]);
            }
          }
        }
      }
    }
  }
}

void GraphNetwork::InitNetwork() {
  for (auto& it : layers_) {
    for (auto& it_n : it->GetNeurons()) {
      for (auto& it_n_w : it_n.GetWeight()) {
        it_n_w = static_cast<double>(std::rand()) / RAND_MAX * 2 - 1;
      }
    }
  }
}

void GraphNetwork::ShowNetwork() {
  std::cout << "Number of layers: " << layers_.size() << std::endl;
  for (auto& it : layers_) {
    std::cout << "Layer type: " << it->GetType();
    std::cout << " Size: " << it->GetNeurons().size() << std::endl;
    std::cout << "Weight: ";
    for (auto& it_n : it->GetNeurons()) {
      std::cout << "[ ";
      for (auto& it_n_w : it_n.GetWeight()) {
        std::cout << it_n_w << " ";
      }
      std::cout << "]";
    }
    std::cout << std::endl;
    std::cout << "Values: ";
    for (auto& it_n : it->GetNeurons()) {
      std::cout << it_n.GetValue() << " ";
    }
    std::cout << std::endl;
    std::cout << "Delta: ";
    for (auto& it_n : it->GetNeurons()) {
      std::cout << it_n.GetDelta() << " ";
    }
    std::cout << std::endl;
    std::cout << "Input neurons: ";
    for (auto& it_n : it->GetNeurons()) {
      std::cout << "{";
      it_n.ShowInputNeurons();
      std::cout << "} ";
    }
    std::cout << std::endl;
  }
}

bool GraphNetwork::TrainNetwork(std::ifstream& fp, size_t& count,
                                size_t g_begin, size_t g_end) {
  for (size_t max = count + kDataSetBatchSize; count < max && !fp.eof();
       ++count) {
    std::string line;
    std::getline(fp, line);
    if (line != "") {
      if (count < g_begin || count > g_end) {
        ReadEmnistLetter(line);
        EmnistLetterToVector_();
        CalculateVector_();
        CalculateDeltaWeights_(emnist_letter_.front());
        UpdateWeights_();
      }
    }
  }
  if (!fp.eof()) {
    return true;
  } else {
    return false;
  }
}

bool GraphNetwork::TestNetwork(std::ifstream& fp, size_t& count,
                               size_t max_tests) {
  for (size_t max = count + kDataSetBatchSize;
       count < max && count <= max_tests && !fp.eof(); ++count) {
    std::string line;
    std::getline(fp, line);
    if (line != "") {
      ReadEmnistLetter(line);
      EmnistLetterToVector_();
      CalculateVector_();
      int max = MaxElement_();
      ++(*confusion_matrix_)(emnist_letter_.front() - 1, max);
      if (emnist_letter_.front() != MaxElement_() + 1) {
        ++count_errors_;
      }
    }
  }
  if (!fp.eof()) {
    return true;
  } else {
    --count;
    return false;
  }
}

void GraphNetwork::EmnistLetterToVector_() {
  vector_.clear();
  for (size_t i = 1; i < emnist_letter_.size(); ++i) {
    vector_.push_back(static_cast<double>(emnist_letter_[i]) / 255.0);
  }
}

void GraphNetwork::CalculateVector_() {
  for (auto& it : layers_) {
    for (auto& it_n : it->GetNeurons()) {
      double sum = 0;
      for (size_t i = 0; i < it_n.GetWeight().size(); ++i) {
        sum += it_n.GetWeight()[i] * vector_[i];
      }
      it_n.GetValue() = Sigmoid_(sum);
    }
    vector_.clear();
    for (auto& it_n : it->GetNeurons()) {
      vector_.push_back(it_n.GetValue());
    }
  }
}

void GraphNetwork::LoadWeights(const std::string& weights_file) {
  std::ifstream fp(weights_file);
  if (fp.is_open()) {
    std::string line;
    std::getline(fp, line);
    if (line == "Network weights:") {
      std::getline(fp, line);
      size_t num_layers = std::stoul(line);
      if (num_layers == layers_.size()) {
        for (auto& it : layers_) {
          it->Load(&fp);
        }
      } else if ((num_layers >= kMinHiddenLayers + 2) &&
                 (num_layers <= kMaxHiddenLayers + 2)) {
        GenerateNetwork(num_layers - 2);
        for (auto& it : layers_) {
          it->Load(&fp);
        }
      } else {
        fp.close();
        throw std::invalid_argument("Error: incorrect format of " +
                                    kWeightsFile);
      }
    } else {
      fp.close();
      throw std::invalid_argument("Error: incorrect format of " + kWeightsFile);
    }
  } else {
    throw std::invalid_argument("Error: can't open the " + kWeightsFile);
  }
}

void GraphNetwork::SaveWeights(const std::string& weights_file) {
  std::ofstream fp(weights_file);
  if (fp.is_open()) {
    fp << "Network weights:" << std::endl;
    fp << layers_.size() << std::endl;
    for (auto& it : layers_) {
      it->Save(&fp);
    }
    fp.close();
  } else {
    throw std::invalid_argument("Error: can't save the " + kWeightsFile);
  }
}

void GraphNetwork::CalculateDeltaWeights_(size_t expected) {
  for (auto rit = layers_.rbegin(); rit != layers_.rend(); rit++) {
    if ((*rit)->GetType() == kOutputLayer) {
      for (size_t i = 0; i < (*rit)->GetNeurons().size(); ++i) {
        double value = (*rit)->GetNeurons()[i].GetValue();
        if (i + 1 == expected) {
          (*rit)->GetNeurons()[i].GetDelta() =
              value * (1 - value) * (1 - value);
        } else {
          (*rit)->GetNeurons()[i].GetDelta() = -value * (1 - value) * value;
        }
      }
    } else {
      rit--;
      auto neurons_prev = (*rit)->GetNeurons();
      rit++;
      for (size_t i = 0; i < (*rit)->GetNeurons().size(); ++i) {
        double value = (*rit)->GetNeurons()[i].GetValue();
        double sum = 0.0;
        for (size_t j = 0; j < neurons_prev.size(); ++j) {
          sum += neurons_prev[j].GetWeight()[i] * neurons_prev[j].GetDelta();
        }
        (*rit)->GetNeurons()[i].GetDelta() = value * (1 - value) * sum;
      }
    }
  }
}

void GraphNetwork::UpdateWeights_() {
  EmnistLetterToVector_();

  for (auto& it : layers_) {
    auto& neurons = it->GetNeurons();
    for (size_t i = 0; i < neurons.size(); ++i) {
      for (size_t j = 0; j < vector_.size(); ++j) {
        neurons[i].GetWeight()[j] +=
            vector_[j] * neurons[i].GetDelta() * learning_rate_;
      }
    }
    vector_.clear();
    for (auto& it_n : it->GetNeurons()) {
      vector_.push_back(it_n.GetValue());
    }
  }
}

int GraphNetwork::Predict(const std::vector<int>& input_layer) {
  vector_.clear();
  for (auto& it : input_layer) {
    vector_.push_back(static_cast<double>(it) / 255.0);
  }
  CalculateVector_();
  int result = static_cast<int>(MaxElement_());
  return result;
}

int GraphNetwork::MaxElement_() {
  if (vector_.empty()) {
    throw std::out_of_range("Error: vector is empty");
  }
  int i = 0;
  int result = 0;
  for (auto& it : vector_) {
    if (vector_[result] < it) {
      result = i;
    }
    ++i;
  }
  return result;
}

//  GraphNetwork::Layer

void GraphNetwork::Layer::Load(std::ifstream* fp) {
  std::string line;
  std::getline(*fp, line, ' ');
  size_t rows = 0, cols = 0;
  try {
    rows = std::stoul(line);
  } catch (const std::exception& e) {
    throw std::out_of_range("Error: index out of range");
  }
  std::getline(*fp, line);
  try {
    cols = std::stoul(line);
  } catch (const std::exception& e) {
    throw std::out_of_range("Error: index out of range");
  }
  if (neurons_.size() == cols && neurons_[0].GetWeight().size() == rows) {
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols - 1; ++j) {
        std::getline(*fp, line, ' ');
        neurons_[j].GetWeight()[i] = std::stod(line);
      }
      std::getline(*fp, line);
      neurons_[cols - 1].GetWeight()[i] = std::stod(line);
    }
  } else {
    throw std::out_of_range("Error: incorrect format");
  }
}

void GraphNetwork::Layer::Save(std::ofstream* fp) {
  *fp << neurons_[0].GetWeight().size() << " " << neurons_.size() << std::endl;
  for (size_t i = 0; i < neurons_[0].GetWeight().size(); ++i) {
    for (size_t j = 0; j < neurons_.size(); ++j) {
      *fp << neurons_[j].GetWeight()[i] << " ";
    }
    *fp << std::endl;
  }
}

}  // namespace s21
