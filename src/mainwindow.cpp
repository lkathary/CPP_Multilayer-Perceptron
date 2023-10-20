#include "mainwindow.h"

#include <QFileDialog>
#include <QGraphicsTextItem>
#include <chrono>  // NOLINT(*)

#include "ui_mainwindow.h"

s21::Controller* s21::Controller::controller_ = nullptr;

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent),
      ui(new Ui::MainWindow),
      draw_dialog_(new DrawDialog),
      scene_(new QGraphicsScene),
      graph_scene_(new QGraphicsScene),
      network_instance_(new s21::MatrixNetwork),
      graph_instance_(new s21::GraphNetwork) {
  ui->setupUi(this);
  this->setFixedSize(this->geometry().width(), this->geometry().height());
  scene_->setSceneRect(0, 0, s21::kNumNeurons * 5, s21::kNumNeurons * 5);
  ui->graphicsViewLetter->setScene(scene_);

  graph_scene_->setSceneRect(0, 0, ui->graphicsViewGraph->width(),
                             ui->graphicsViewGraph->height());
  ui->graphicsViewGraph->setScene(graph_scene_);

  s21::Controller* ctrl = s21::Controller::GetInstance();
  ctrl->Connect(network_instance_, graph_instance_);
}

MainWindow::~MainWindow() {
  delete ui;
  delete draw_dialog_;
  delete scene_;
  delete graph_scene_;
  delete network_instance_;
  delete graph_instance_;
  s21::Controller* ctrl = s21::Controller::GetInstance();
  delete ctrl;
}

void MainWindow::on_pushButtonInfo_clicked() {
  s21::Controller* ctrl = s21::Controller::GetInstance();
  QString str = "Net type: ";
  if (ctrl->GetType() == s21::kMatrixNet) {
    str += "Matrix";
  } else {
    str += "Graph";
  }
  ui->textInfo->append(str);
  str = "Number of layers: " + QString::number(ctrl->GetNumLayers());
  ui->textInfo->append(str);
  str = "Input layer: " + QString::number(ctrl->GetInputLayerNeurons()) +
        " neurons";
  ui->textInfo->append(str);
  str = "Hidden layers: " + QString::number(ctrl->GetHiddenLayerNeurons()) +
        " neurons";
  ui->textInfo->append(str);
  str = "Output layers: " + QString::number(ctrl->GetOutputLayerNeurons()) +
        " neurons";
  ui->textInfo->append(str);

  ui->textInfo->append("\nTrain file:  " +
                       QString::fromStdString(s21::kDataSetTrain));
  ui->textInfo->append("Test file:  " +
                       QString::fromStdString(s21::kDataSetTest));

  str = "\nEmnistLetter:";
  ui->textInfo->append(str);
  str = "";
  auto EmnistLetter = ctrl->GetEmnistLetter();
  int i = -1;
  for (auto it : EmnistLetter) {
    if (!(++i)) {
      str = "Symbol {" + QString::number(it) + "}";
    } else {
      str += ", " + QString::number(it);
    }
  }
  ui->textInfo->append(str);
}

void MainWindow::on_pushButtonClear_clicked() { ui->textInfo->clear(); }

void MainWindow::on_pushButtonSaveNet_clicked() {
  QString fileName;
  fileName = QFileDialog::getSaveFileName(this, tr("Save Network"), "",
                                          tr("NetWork Files (*.txt)"));
  if (!fileName.isNull()) {
    try {
      s21::Controller* ctrl = s21::Controller::GetInstance();
      ctrl->SaveWeights(fileName.toStdString());
      ui->textInfo->append("File:  " + QFileInfo(fileName).fileName() +
                           " saved");
    } catch (const std::exception& e) {
      ui->textInfo->append("File saving error");
    }
  }
}

void MainWindow::on_pushButtonOpenNet_clicked() {
  QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"), ".",
                                                  tr("txt files (*.txt)"));
  if (!fileName.isNull()) {
    s21::Controller* ctrl = s21::Controller::GetInstance();
    std::string result = ctrl->LoadWeights(fileName.toStdString());
    ui->textInfo->append(QString::fromStdString(result));
  }
}

void MainWindow::on_pushButtonGenerateNet_clicked() {
  s21::Controller* ctrl = s21::Controller::GetInstance();
  if (ctrl->GetType() == s21::kMatrixNet) {
    ui->textInfo->append("Generating MatrixNet...");
  } else {
    ui->textInfo->append("Generating GraphNet...");
  }
  ctrl->GenerateNetwork(ui->spinBoxNumHiddenLayers->value());
  ui->textInfo->append("Done");
}

void MainWindow::on_pushButtonTrain_clicked() {
  DisableUI_();
  s21::Controller* ctrl = s21::Controller::GetInstance();
  ctrl->SetLearningRate(ui->LearningRate->value());
  ui->textInfo->append(
      "=== Train for epochs: " + ui->LearningEpoch->cleanText() +
      " with LearningRate: " + ui->LearningRate->cleanText() + " ===");
  error_.clear();
  graph_scene_->clear();
  std::ifstream fp(s21::kDataSetTrain);
  if (fp.is_open()) {
    for (int epoch = 1; epoch <= ui->LearningEpoch->value(); ++epoch) {
      size_t g_begin = 0;
      size_t g_end = 0;
      if (ui->checkBoxCrossValidation->isChecked()) {
        g_begin = (epoch - 1) * s21::kNumDataSetSamples /
                      ui->LearningGroups->value() +
                  1;
        g_end = epoch * s21::kNumDataSetSamples / ui->LearningGroups->value();
        ui->textInfo->append("G_begin: " + QString::number(g_begin) +
                             " G_end: " + QString::number(g_end));
      }
      size_t count = 1;
      for (; ctrl->TrainNetwork(fp, count, g_begin, g_end) &&
             count <= s21::kNumDataSetSamples;) {
        ui->textInfo->append(
            QString::number(count - 1) +
            " samples processed (epoch: " + QString::number(epoch) + ")");
        QApplication::processEvents();
      }
      ui->textInfo->append(
          QString::number(count - 1) +
          " samples processed (epoch: " + QString::number(epoch) + ")");

      std::ifstream fp_test(s21::kDataSetTest);
      if (fp_test.is_open()) {
        ctrl->ResetStatistics();
        size_t count_test = 1;
        ui->textInfo->append("=== " + QString::number(s21::kNumDataSetTests) +
                             " Tests ===");
        for (; ctrl->TestNetwork(fp_test, count_test, s21::kNumDataSetTests) &&
               count_test <= s21::kNumDataSetTests;) {
          QApplication::processEvents();
        }
        fp_test.close();
      }
      error_.push_back(1 - ctrl->CalculateAccuracy() / 100);
      fp.clear();
      fp.seekg(0, std::ios_base::beg);
    }

    ui->textInfo->append("Done");
    fp.close();
  }
  for (auto& it : error_) {
    ui->textInfo->append("Error: " + QString::number(it));
  }
  DrawGraph_();
  EnableUI_();
}

void MainWindow::on_pushButtonTest_clicked() {
  DisableUI_();
  s21::Controller* ctrl = s21::Controller::GetInstance();
  std::ifstream fp(s21::kDataSetTest);
  if (fp.is_open()) {
    ctrl->ResetStatistics();
    size_t count = 1;
    size_t max_tests = static_cast<size_t>(s21::kNumDataSetTests *
                                           ui->BoxPartTests->value() / 100);
    ui->textInfo->append("=== " + QString::number(max_tests) + " Tests ===");
    auto begin = std::chrono::high_resolution_clock::now();
    for (; ctrl->TestNetwork(fp, count, max_tests) && count <= max_tests;) {
      ui->textInfo->append(QString::number(count - 1) + " tests processed");
      ui->textInfo->append(QString::number(ctrl->GetCountErrors()) +
                           " tests failed");
      QApplication::processEvents();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - begin;
    fp.close();

    ui->textInfo->append(QString::number(count - 1) + " tests processed");
    ui->textInfo->append(QString::number(ctrl->GetCountErrors()) +
                         " tests failed");
    ui->textInfo->append("Done");

    ui->labelAccuracy->setText(
        QString::number(ctrl->CalculateAccuracy(), 'g', 4) + " %");
    ui->labelPrecision->setText(
        QString::number(ctrl->CalculatePrecision(), 'g', 4) + " %");
    ui->labelRecall->setText(QString::number(ctrl->CalculateRecall(), 'g', 4) +
                             " %");
    ui->labelFmeasure->setText(
        QString::number(ctrl->CalculateFmeasure(), 'g', 4) + " %");
    ui->labelTimeSpent->setText(QString::number(duration.count(), 'g', 4) +
                                " s");
  }
  EnableUI_();
}

void MainWindow::on_radioButtonMatrix_clicked() {
  s21::Controller* ctrl = s21::Controller::GetInstance();
  ctrl->SetCurrentNetwork(s21::kMatrixNet);
}

void MainWindow::on_radioButtonGraph_clicked() {
  s21::Controller* ctrl = s21::Controller::GetInstance();
  ctrl->SetCurrentNetwork(s21::kGraphNet);
}

void MainWindow::on_BoxPartTests_valueChanged(double value) {
  ui->SliderPartTests->setValue(value);
}

void MainWindow::on_SliderPartTests_valueChanged(int value) {
  ui->BoxPartTests->setValue(value);
}

void MainWindow::on_LearningEpoch_valueChanged(int value) {
  ui->LearningGroups->setMinimum(value);
}

void MainWindow::on_checkBoxCrossValidation_clicked(bool checked) {
  ui->labelGroups->setEnabled(checked);
  ui->LearningGroups->setEnabled(checked);
}

void MainWindow::on_pushButtonGetImage_clicked() {
  draw_dialog_->exec();
  ImageRecognition_();
}

void MainWindow::on_pushButtonLoadImage_clicked() {
  QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"), ".",
                                                  tr("Image files (*.bmp)"));
  if (!fileName.isNull()) {
    QImage image(fileName);
    *(draw_dialog_->GetImage()) =
        image.scaled(s21::kSizeImage, s21::kSizeImage, Qt::IgnoreAspectRatio,
                     Qt::SmoothTransformation);
    ui->textInfo->append("File:  " + QFileInfo(fileName).fileName() +
                         " loaded");
    ImageRecognition_();
  }
}

void MainWindow::ImageRecognition_() {
  QImage scaled_image(draw_dialog_->GetImage()->scaled(
      s21::kNumNeurons * 5, s21::kNumNeurons * 5, Qt::IgnoreAspectRatio,
      Qt::SmoothTransformation));
  scene_->addPixmap(QPixmap::fromImage(scaled_image));

  QImage real_image(draw_dialog_->GetImage()->scaled(
      s21::kNumNeurons, s21::kNumNeurons, Qt::IgnoreAspectRatio,
      Qt::SmoothTransformation));
  std::vector<int> input_layer;
  for (int i = 0; i < s21::kNumNeurons; ++i) {
    for (int j = 0; j < s21::kNumNeurons; ++j) {
      input_layer.push_back(real_image.pixelColor(i, j).black());
    }
  }
  s21::Controller* ctrl = s21::Controller::GetInstance();
  char result = static_cast<char>(ctrl->Predict(input_layer) + 65);
  ui->textInfo->append("Prediction: " + QString(result));
  ui->labelResult->setText(QString(result));
  // std::pair<int, double> result = ctrl->Predict(input_layer);
  // ui->textInfo->append("Prediction: " +
  // QString(static_cast<char>(result.first + 65)) + " "
  //                                     + QString::number(result.second * 100,
  //                                     'g', 4) + "%");
}

void MainWindow::DrawGraph_() {
  const int kDelta = 10;
  auto height = graph_scene_->height();
  auto width = graph_scene_->width();
  graph_scene_->addLine(kDelta, kDelta, kDelta, height);
  graph_scene_->addLine(0, height - kDelta, width - kDelta, height - kDelta);
  auto delta_x = (width - 2 * kDelta) / error_.size();
  for (size_t i = 1; i <= error_.size(); ++i) {
    graph_scene_->addLine(delta_x * i, height - kDelta, delta_x * i,
                          (height - kDelta) * (1 - error_[i - 1]),
                          QPen(Qt::blue, 5, Qt::SolidLine, Qt::SquareCap));
    QGraphicsTextItem* text =
        graph_scene_->addText(QString::number(error_[i - 1], 'g', 2));
    text->setPos(delta_x * i - 2 * kDelta,
                 (height - kDelta) * (1 - error_[i - 1]) - 3 * kDelta);
  }
}

void MainWindow::EnableUI_() {
  ui->pushButtonGenerateNet->setEnabled(true);
  ui->pushButtonOpenNet->setEnabled(true);
  ui->pushButtonSaveNet->setEnabled(true);
  ui->pushButtonTrain->setEnabled(true);
  ui->pushButtonTest->setEnabled(true);
  ui->pushButtonGetImage->setEnabled(true);
  ui->pushButtonLoadImage->setEnabled(true);
  ui->pushButtonInfo->setEnabled(true);

  ui->radioButtonMatrix->setEnabled(true);
  ui->radioButtonGraph->setEnabled(true);
  ui->spinBoxNumHiddenLayers->setEnabled(true);

  ui->LearningRate->setEnabled(true);
  ui->LearningEpoch->setEnabled(true);
  ui->checkBoxCrossValidation->setEnabled(true);
  ui->LearningGroups->setEnabled(true);

  ui->SliderPartTests->setEnabled(true);
  ui->BoxPartTests->setEnabled(true);
}

void MainWindow::DisableUI_() {
  ui->pushButtonGenerateNet->setEnabled(false);
  ui->pushButtonOpenNet->setEnabled(false);
  ui->pushButtonSaveNet->setEnabled(false);
  ui->pushButtonTrain->setEnabled(false);
  ui->pushButtonTest->setEnabled(false);
  ui->pushButtonGetImage->setEnabled(false);
  ui->pushButtonLoadImage->setEnabled(false);
  ui->pushButtonInfo->setEnabled(false);

  ui->radioButtonMatrix->setEnabled(false);
  ui->radioButtonGraph->setEnabled(false);
  ui->spinBoxNumHiddenLayers->setEnabled(false);

  ui->LearningRate->setEnabled(false);
  ui->LearningEpoch->setEnabled(false);
  ui->checkBoxCrossValidation->setEnabled(false);
  ui->LearningGroups->setEnabled(false);

  ui->SliderPartTests->setEnabled(false);
  ui->BoxPartTests->setEnabled(false);
}
