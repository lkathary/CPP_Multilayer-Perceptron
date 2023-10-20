#include "matrix.h"

namespace s21 {

Matrix::Matrix(int rows, int cols) : matrix_(nullptr) {
  if (rows < 1 || cols < 1) {
    throw std::out_of_range("Error: rows or columns < 1");
  }
  AllocateMem(rows, cols);
}

Matrix::Matrix(Matrix&& other) {
  Clear();
  std::swap(rows_, other.rows_);
  std::swap(cols_, other.cols_);
  std::swap(matrix_, other.matrix_);
}

void Matrix::MulMatrix(const Matrix& other) {
  if (cols_ != other.rows_) {
    throw std::range_error("Error: incompatible matrix dimensions");
  }
  Matrix result(rows_, other.cols_);
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < other.cols_; ++j) {
      for (int k = 0; k < this->cols_; ++k) {
        result.matrix_[i][j] += this->matrix_[i][k] * other.matrix_[k][j];
      }
    }
  }
  *this = result;
}

void Matrix::MulMatrixWithSigmoid(const Matrix& other) {
  if (cols_ != other.rows_) {
    throw std::range_error("Error: incompatible matrix dimensions");
  }
  Matrix result(rows_, other.cols_);
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < other.cols_; ++j) {
      for (int k = 0; k < this->cols_; k++) {
        result.matrix_[i][j] += this->matrix_[i][k] * other.matrix_[k][j];
      }
      result.matrix_[i][j] = Sigmoid(result.matrix_[i][j]);
    }
  }
  *this = result;
}

void Matrix::RandomizeMatrix() {
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      matrix_[i][j] = static_cast<double>(std::rand()) / RAND_MAX * 2 - 1;
    }
  }
}

Matrix& Matrix::operator=(const Matrix& other) {
  if (this != &other) {
    Clear();
    AllocateMem(other.rows_, other.cols_);
    for (int i = 0; i < rows_; ++i) {
      for (int j = 0; j < cols_; ++j) {
        this->matrix_[i][j] = other.matrix_[i][j];
      }
    }
  }
  return *this;
}

Matrix Matrix::operator*(const Matrix& other) {
  Matrix result(*this);
  result.MulMatrix(other);
  return result;
}

double& Matrix::operator()(int row, int col) {
  if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
    throw std::out_of_range("Error: index out of range");
  }
  return matrix_[row][col];
}

void Matrix::Save(std::ofstream* fp) {
  *fp << rows_ << " " << cols_ << std::endl;
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      *fp << matrix_[i][j] << " ";
    }
    *fp << std::endl;
  }
}

void Matrix::Load(std::ifstream* fp) {
  std::string line;
  std::getline(*fp, line, ' ');
  int rows = 0, cols = 0;
  try {
    rows = std::stoi(line);
  } catch (const std::exception& e) {
    throw std::out_of_range("Error: index out of range");
  }
  std::getline(*fp, line);
  try {
    cols = std::stoi(line);
  } catch (const std::exception& e) {
    throw std::out_of_range("Error: index out of range");
  }
  if (rows_ == rows && cols_ == cols) {
    for (int i = 0; i < rows_; ++i) {
      for (int j = 0; j < cols_ - 1; ++j) {
        std::getline(*fp, line, ' ');
        matrix_[i][j] = std::stod(line);
      }
      std::getline(*fp, line);
      matrix_[i][cols_ - 1] = std::stod(line);
    }
  } else {
    throw std::out_of_range("Error: incorrect format");
  }
}

void Matrix::Show() {
  std::cout << "matrix(" << rows_ << ":" << cols_ << ") [" << this << ":"
            << matrix_ << "]" << std::endl;
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      std::cout << "[" << i << "," << j << "]=";
      std::cout << matrix_[i][j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

int Matrix::MaxElement() {
  if (matrix_ == nullptr) {
    throw std::out_of_range("Error: matrix is empty (nullptr)");
  }
  int result = 0;
  for (int j = 0; j < cols_; ++j) {
    if (matrix_[0][result] < matrix_[0][j]) {
      result = j;
    }
  }
  return result;
}

void Matrix::AllocateMem(int rows, int cols) {
  rows_ = rows;
  cols_ = cols;
  matrix_ = new double*[rows_];
  for (int i = 0; i < rows_; i++) {
    matrix_[i] = new double[cols_]();
  }
}

void Matrix::Clear() {
  if (matrix_) {
    for (int i = 0; i < rows_; ++i) {
      delete[] matrix_[i];
    }
    delete[] matrix_;
    matrix_ = nullptr;
  }
}

}  // namespace s21
