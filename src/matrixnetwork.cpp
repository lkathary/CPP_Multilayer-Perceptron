#include "matrixnetwork.h"

namespace s21 {

MatrixNetwork::MatrixNetwork() : MatrixNetwork(kNumHiddenLayers) {}

MatrixNetwork::MatrixNetwork(int num_hidden_layers) {
  srand(time(0));
  GenerateNetwork(num_hidden_layers);
}

MatrixNetwork::~MatrixNetwork() {
  for (auto& it : layers_) {
    delete it;
  }
}

void MatrixNetwork::Clear() {
  for (auto& it : layers_) {
    delete it;
  }
  layers_.clear();
}

void MatrixNetwork::GenerateNetwork(int num_hidden_layers) {
  Clear();
  layers_.push_back(new Layer(kInputLayer));
  for (int i = 0; i < num_hidden_layers; ++i) {
    layers_.push_back(new Layer(kHiddenLayer));
  }
  layers_.push_back(new Layer(kOutputLayer));
}

void MatrixNetwork::InitNetwork() {
  for (auto& it : layers_) {
    it->GetMatrix()->RandomizeMatrix();
  }
}

void MatrixNetwork::ShowNetwork() {
  for (auto& it : layers_) {
    std::cout << "Weights: " << std::endl;
    it->GetMatrix()->Show();
    std::cout << "Vector: " << std::endl;
    it->GetVector()->Show();
    std::cout << "Delta: " << std::endl;
    it->GetDelta()->Show();
  }
}

bool MatrixNetwork::TrainNetwork(std::ifstream& fp, size_t& count,
                                 size_t g_begin, size_t g_end) {
  for (size_t max = count + kDataSetBatchSize; count < max && !fp.eof();
       ++count) {
    std::string line;
    std::getline(fp, line);
    if (line != "") {
      if (count < g_begin || count > g_end) {
        Matrix* vector;
        ReadEmnistLetter(line);
        vector = EmnistLetterToVector_();
        CalculateVector_(vector);
        CalculateDeltaWeights_(emnist_letter_.front());
        UpdateWeights_();
        delete vector;
      }
    }
  }
  if (!fp.eof()) {
    return true;
  } else {
    return false;
  }
}

bool MatrixNetwork::TestNetwork(std::ifstream& fp, size_t& count,
                                size_t max_tests) {
  for (size_t max = count + kDataSetBatchSize;
       count < max && count <= max_tests && !fp.eof(); ++count) {
    std::string line;
    std::getline(fp, line);
    if (line != "") {
      Matrix* vector;
      ReadEmnistLetter(line);
      vector = EmnistLetterToVector_();
      CalculateVector_(vector);
      int max = vector->MaxElement();
      ++(*confusion_matrix_)(emnist_letter_.front() - 1, max);
      if (emnist_letter_.front() != vector->MaxElement() + 1) {
        ++count_errors_;
      }
      delete vector;
    }
  }
  if (!fp.eof()) {
    return true;
  } else {
    --count;
    return false;
  }
}

Matrix* MatrixNetwork::EmnistLetterToVector_() {
  Matrix* vec = new Matrix(1, kInputLayerNeurons);
  int i = -1;
  for (auto& it : emnist_letter_) {
    if (i != -1) {
      (*vec)(0, i++) = static_cast<double>(it) / 255.0;
    } else {
      ++i;
    }
  }
  return vec;
}

void MatrixNetwork::CalculateVector_(Matrix* vector) {
  for (auto& it : layers_) {
    vector->MulMatrixWithSigmoid(*(it->GetMatrix()));
    *(it->GetVector()) = *vector;
  }
}

void MatrixNetwork::SaveWeights(const std::string& weights_file) {
  std::ofstream fp(weights_file);
  if (fp.is_open()) {
    fp << "Network weights:" << std::endl;
    fp << layers_.size() << std::endl;
    for (auto& it : layers_) {
      it->GetMatrix()->Save(&fp);
    }
    fp.close();
  } else {
    throw std::invalid_argument("Error: can't save the " + kWeightsFile);
  }
}

void MatrixNetwork::LoadWeights(const std::string& weights_file) {
  std::ifstream fp(weights_file);
  if (fp.is_open()) {
    std::string line;
    std::getline(fp, line);
    if (line == "Network weights:") {
      std::getline(fp, line);
      size_t num_layers = std::stoul(line);
      if (num_layers == layers_.size()) {
        for (auto& it : layers_) {
          it->GetMatrix()->Load(&fp);
        }
      } else if ((num_layers >= kMinHiddenLayers + 2) &&
                 (num_layers <= kMaxHiddenLayers + 2)) {
        GenerateNetwork(num_layers - 2);
        for (auto& it : layers_) {
          it->GetMatrix()->Load(&fp);
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

void MatrixNetwork::CalculateDeltaWeights_(int expected) {
  for (auto rit = layers_.rbegin(); rit != layers_.rend(); rit++) {
    if ((*rit)->GetType() == kOutputLayer) {
      for (int j = 0; j < (*((*rit)->GetDelta())).GetCols(); ++j) {
        double value = (*(*rit)->GetVector())(0, j);
        if (j + 1 == expected) {
          (*(*rit)->GetDelta())(0, j) = value * (1 - value) * (1 - value);
        } else {
          (*(*rit)->GetDelta())(0, j) = -value * (1 - value) * value;
        }
      }
    } else {
      rit--;
      auto weights_prev = *((*rit)->GetMatrix());
      auto delta_weights_prev = *((*rit)->GetDelta());
      rit++;
      for (int j = 0; j < (*((*rit)->GetDelta())).GetCols(); ++j) {
        double value = (*(*rit)->GetVector())(0, j);
        double sum = 0.0;
        for (int k = 0; k < delta_weights_prev.GetCols(); ++k) {
          sum += weights_prev(j, k) * delta_weights_prev(0, k);
        }
        (*(*rit)->GetDelta())(0, j) = value * (1 - value) * sum;
      }
    }
  }
}

void MatrixNetwork::UpdateWeights_() {
  Matrix vector_prev = *EmnistLetterToVector_();

  for (auto& it : layers_) {
    for (int i = 0; i < it->GetMatrix()->GetRows(); ++i) {
      for (int j = 0; j < it->GetMatrix()->GetCols(); ++j) {
        (*it->GetMatrix())(i, j) +=
            vector_prev(0, i) * (*it->GetDelta())(0, j) * learning_rate_;
      }
    }
    vector_prev = *(it->GetVector());
  }
}

int MatrixNetwork::Predict(const std::vector<int>& input_layer) {
  Matrix* vector = new Matrix(1, kInputLayerNeurons);
  int i = 0;
  for (auto& it : input_layer) {
    (*vector)(0, i++) = static_cast<double>(it) / 255.0;
  }
  CalculateVector_(vector);
  int result = vector->MaxElement();
  delete vector;
  return result;
}

// std::pair<int, double> MatrixNetwork::Predict(const std::vector<int>&
// input_layer) {
//     Matrix* vector = new Matrix(1, kInputLayerNeurons);
//     int i = 0;
//     for (auto& it : input_layer) {
//         (*vector)(0, i++) = static_cast<double>(it) / 255.0;
//     }
//     CalculateVector_(vector);
//     int max = vector->MaxElement();
//     std::pair<int, double> result = std::make_pair(max, (*vector)(0, max));
//     delete vector;
//     return result;
// }

}  // namespace s21
