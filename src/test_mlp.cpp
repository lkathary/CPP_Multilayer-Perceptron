#include <gtest/gtest.h>

#include "graphnetwork.h"
#include "matrix.h"
#include "matrixnetwork.h"

namespace s21 {

const std::string kWeightsFileLoad = "./weights/weights_2_784_86__.txt";
const std::string kWeightsFileSave = "./weights/weights_2_784_test.txt";

}  // namespace s21

constexpr double kEPS = 1e-7;

TEST(Matrix, Create) {
  s21::Matrix one_instance(2, 3);
  one_instance(0, 0) = 1;
  one_instance(1, 1) = 2;
  one_instance(1, 2) = 5.15;
  ASSERT_EQ(one_instance.GetRows(), 2);
  ASSERT_EQ(one_instance.GetCols(), 3);
  ASSERT_NEAR(one_instance(1, 2), 5.15, kEPS);
}

TEST(Matrix, Mul) {
  s21::Matrix one_instance(2, 2);
  one_instance(0, 0) = 1.0;
  one_instance(0, 1) = -2.0;
  one_instance(1, 0) = 5.15;
  one_instance(1, 1) = -0.2;

  s21::Matrix two_instance(1, 2);
  two_instance(0, 0) = 1.0;
  two_instance(0, 1) = 2.0;
  two_instance.MulMatrix(one_instance);

  ASSERT_NEAR(two_instance(0, 0), 11.3, kEPS);
  ASSERT_NEAR(two_instance(0, 1), -2.4, kEPS);
}

TEST(Matrix, Mul_via_operator_mul) {
  s21::Matrix one_instance(2, 2);
  one_instance(0, 0) = 1.0;
  one_instance(0, 1) = -2.0;
  one_instance(1, 0) = 5.15;
  one_instance(1, 1) = -0.2;

  s21::Matrix two_instance(1, 2);
  two_instance(0, 0) = 1.0;
  two_instance(0, 1) = 2.0;
  two_instance = two_instance * one_instance;

  ASSERT_NEAR(two_instance(0, 0), 11.3, kEPS);
  ASSERT_NEAR(two_instance(0, 1), -2.4, kEPS);
}

TEST(Matrix, MulMatrixWithSigmoid) {
  s21::Matrix one_instance(2, 2);
  one_instance(0, 0) = 1.0;
  one_instance(0, 1) = -2.0;
  one_instance(1, 0) = 5.15;
  one_instance(1, 1) = -0.2;

  s21::Matrix two_instance(1, 2);
  two_instance(0, 0) = 1.0;
  two_instance(0, 1) = 2.0;
  two_instance.MulMatrixWithSigmoid(one_instance);

  ASSERT_NEAR(two_instance(0, 0), 0.9999876, kEPS);
  ASSERT_NEAR(two_instance(0, 1), 0.0831727, kEPS);
}

TEST(Matrix, MaxElement) {
  s21::Matrix one_instance(1, 4);
  one_instance(0, 0) = 1.0;
  one_instance(0, 1) = -2.0;
  one_instance(0, 2) = 5.15;
  one_instance(0, 3) = -0.2;

  ASSERT_EQ(one_instance.MaxElement(), 2);
}

TEST(Matrix, Extra) {
  s21::Matrix one_instance(3, 5);
  one_instance.RandomizeMatrix();
  one_instance.Show();
  ASSERT_TRUE(true);
}

TEST(Network, Statistic) {
  s21::MatrixNetwork mn;
  mn.LoadWeights(s21::kWeightsFileLoad);
  mn.ResetStatistics();
  mn.CalculateAccuracy();
  mn.CalculatePrecision();
  mn.CalculateRecall();
  mn.CalculateFmeasure();
  std::string line = "1,0,3,0,33,0,0,0,255,14,0,0,0,0,12,0,0";
  ASSERT_THROW(mn.ReadEmnistLetter(line), std::length_error);
}

TEST(MatrixNetwork, Init) {
  s21::MatrixNetwork mn;
  mn.InitNetwork();
  // mn.ShowNetwork();
  ASSERT_TRUE(true);
}

TEST(MatrixNetwork, Load) {
  s21::MatrixNetwork mn;
  mn.LoadWeights(s21::kWeightsFileLoad);
  ASSERT_TRUE(true);
}

TEST(MatrixNetwork, Save) {
  s21::MatrixNetwork mn;
  mn.InitNetwork();
  mn.SaveWeights(s21::kWeightsFileSave);
  ASSERT_TRUE(true);
}

TEST(MatrixNetwork, Predict) {
  s21::MatrixNetwork mn;
  mn.LoadWeights(s21::kWeightsFileLoad);
  std::vector<int> input_vector = {
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   10,
      3,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   132, 252, 226, 46,
      0,   0,   0,   0,   0,   0,   0,   0,   4,   162, 243, 170, 7,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   65,  254, 255, 255, 164, 0,   0,
      0,   0,   0,   0,   0,   0,   81,  255, 255, 255, 131, 0,   0,   0,   0,
      0,   0,   0,   0,   0,   2,   209, 255, 255, 255, 147, 0,   0,   0,   0,
      0,   0,   0,   0,   77,  255, 255, 255, 246, 29,  0,   0,   0,   0,   0,
      0,   0,   0,   86,  255, 255, 255, 241, 29,  0,   0,   0,   0,   0,   0,
      0,   0,   4,   203, 255, 255, 255, 163, 0,   0,   0,   0,   0,   0,   0,
      0,   201, 255, 255, 255, 120, 0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   58,  254, 255, 255, 253, 40,  0,   0,   0,   0,   0,   0,   28,  252,
      255, 255, 239, 12,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      169, 255, 255, 255, 155, 0,   0,   0,   0,   0,   0,   121, 255, 255, 255,
      161, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   42,  253,
      255, 255, 249, 31,  0,   0,   0,   0,   6,   224, 255, 255, 255, 62,  0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   177, 255, 255,
      255, 142, 0,   0,   0,   0,   94,  255, 255, 255, 216, 2,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   52,  255, 255, 255, 241,
      20,  0,   0,   5,   215, 255, 255, 255, 103, 0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   191, 255, 255, 255, 131, 0,
      0,   95,  255, 255, 255, 230, 7,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   69,  255, 255, 255, 238, 16,  4,   218,
      255, 255, 255, 112, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   1,   198, 255, 255, 255, 145, 99,  255, 255, 255,
      227, 10,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   70,  255, 255, 255, 253, 230, 255, 255, 255, 105, 0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   183, 255, 255, 255, 255, 255, 255, 229, 8,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   32,  245, 255, 255, 255, 255, 255, 123, 0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      126, 255, 255, 255, 255, 250, 27,  0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   15,  243,
      255, 255, 255, 181, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   158, 255, 255,
      255, 102, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   102, 255, 255, 255, 51,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   6,   146, 206, 112, 0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0};

  ASSERT_EQ(input_vector.size(), 784);
  ASSERT_EQ(mn.Predict(input_vector), 23);  //  23 == 'V'
}

TEST(GraphNetwork, Init) {
  s21::GraphNetwork gn;
  gn.InitNetwork();
  // gn.ShowNetwork();
  ASSERT_TRUE(true);
}

TEST(GraphNetwork, Load) {
  s21::GraphNetwork gn;
  gn.LoadWeights(s21::kWeightsFileLoad);
  ASSERT_TRUE(true);
}

TEST(GraphNetwork, Save) {
  s21::GraphNetwork gn;
  gn.InitNetwork();
  gn.SaveWeights(s21::kWeightsFileSave);
  ASSERT_TRUE(true);
}

TEST(GraphNetwork, Predict) {
  s21::GraphNetwork gn;
  gn.LoadWeights(s21::kWeightsFileLoad);
  std::vector<int> input_vector = {
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   10,
      3,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   132, 252, 226, 46,
      0,   0,   0,   0,   0,   0,   0,   0,   4,   162, 243, 170, 7,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   65,  254, 255, 255, 164, 0,   0,
      0,   0,   0,   0,   0,   0,   81,  255, 255, 255, 131, 0,   0,   0,   0,
      0,   0,   0,   0,   0,   2,   209, 255, 255, 255, 147, 0,   0,   0,   0,
      0,   0,   0,   0,   77,  255, 255, 255, 246, 29,  0,   0,   0,   0,   0,
      0,   0,   0,   86,  255, 255, 255, 241, 29,  0,   0,   0,   0,   0,   0,
      0,   0,   4,   203, 255, 255, 255, 163, 0,   0,   0,   0,   0,   0,   0,
      0,   201, 255, 255, 255, 120, 0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   58,  254, 255, 255, 253, 40,  0,   0,   0,   0,   0,   0,   28,  252,
      255, 255, 239, 12,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      169, 255, 255, 255, 155, 0,   0,   0,   0,   0,   0,   121, 255, 255, 255,
      161, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   42,  253,
      255, 255, 249, 31,  0,   0,   0,   0,   6,   224, 255, 255, 255, 62,  0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   177, 255, 255,
      255, 142, 0,   0,   0,   0,   94,  255, 255, 255, 216, 2,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   52,  255, 255, 255, 241,
      20,  0,   0,   5,   215, 255, 255, 255, 103, 0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   191, 255, 255, 255, 131, 0,
      0,   95,  255, 255, 255, 230, 7,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   69,  255, 255, 255, 238, 16,  4,   218,
      255, 255, 255, 112, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   1,   198, 255, 255, 255, 145, 99,  255, 255, 255,
      227, 10,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   70,  255, 255, 255, 253, 230, 255, 255, 255, 105, 0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   183, 255, 255, 255, 255, 255, 255, 229, 8,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   32,  245, 255, 255, 255, 255, 255, 123, 0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      126, 255, 255, 255, 255, 250, 27,  0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   15,  243,
      255, 255, 255, 181, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   158, 255, 255,
      255, 102, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   102, 255, 255, 255, 51,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   6,   146, 206, 112, 0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0};

  ASSERT_EQ(input_vector.size(), 784);
  ASSERT_EQ(gn.Predict(input_vector), 23);  //  23 == 'V'
}

int main(int argc, char *argv[]) {
  s21::Matrix one_instance(3, 5);
  one_instance.RandomizeMatrix();
  one_instance.Show();

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
