#ifndef SRC_MAINWINDOW_H_
#define SRC_MAINWINDOW_H_

#include <QGraphicsScene>
#include <QMainWindow>

#include "controller.h"
#include "drawdialog.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
  Q_OBJECT

 public:
  explicit MainWindow(QWidget* parent = nullptr);
  ~MainWindow();

 private slots:
  void on_pushButtonInfo_clicked();
  void on_pushButtonClear_clicked();

  void on_pushButtonSaveNet_clicked();
  void on_pushButtonOpenNet_clicked();
  void on_pushButtonGenerateNet_clicked();

  void on_pushButtonTrain_clicked();
  void on_pushButtonTest_clicked();

  void on_radioButtonMatrix_clicked();
  void on_radioButtonGraph_clicked();
  void on_BoxPartTests_valueChanged(double value);
  void on_SliderPartTests_valueChanged(int value);
  void on_LearningEpoch_valueChanged(int value);
  void on_checkBoxCrossValidation_clicked(bool checked);

  void on_pushButtonGetImage_clicked();
  void on_pushButtonLoadImage_clicked();

 private:
  Ui::MainWindow* ui;
  DrawDialog* draw_dialog_;
  QGraphicsScene* scene_;
  QGraphicsScene* graph_scene_;
  s21::MatrixNetwork* network_instance_;
  s21::GraphNetwork* graph_instance_;
  std::vector<double> error_;

  void ImageRecognition_();
  void DrawGraph_();
  void EnableUI_();
  void DisableUI_();
};

#endif  //  SRC_MAINWINDOW_H_
