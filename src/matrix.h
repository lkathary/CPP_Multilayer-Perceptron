#ifndef SRC_MATRIX_H_
#define SRC_MATRIX_H_

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace s21 {

class Matrix {
 public:
  Matrix() : Matrix(1, 1) {}
  Matrix(int rows, int cols);
  Matrix(const Matrix& other) : matrix_(nullptr) { *this = other; }
  Matrix(Matrix&& other);
  ~Matrix() { Clear(); }

  int GetRows() const { return rows_; }
  int GetCols() const { return cols_; }

  void MulMatrix(const Matrix& other);
  void MulMatrixWithSigmoid(const Matrix& other);
  void RandomizeMatrix();

  Matrix& operator=(const Matrix& other);
  Matrix operator*(const Matrix& other);
  double& operator()(int row, int col);

  void Save(std::ofstream* fp);
  void Load(std::ifstream* fp);
  void Show();
  int MaxElement();

 private:
  int rows_, cols_;
  double** matrix_;

  void AllocateMem(int rows, int cols);
  void Clear();

  double Sigmoid(double value) { return (1.0 / (1.0 + exp(-value))); }
  double DerivativeSigmoid(double value) {
    return Sigmoid(value) * (1.0 - Sigmoid(value));
  }
};

}  // namespace s21

#endif  //  SRC_MATRIX_H_
