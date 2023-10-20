#ifndef SRC_DRAWDIALOG_H_
#define SRC_DRAWDIALOG_H_

#include <QDialog>
#include <QImage>
#include <QMouseEvent>
#include <QPainter>

namespace Ui {
class DrawDialog;
}

class DrawDialog : public QDialog {
  Q_OBJECT

 public:
  explicit DrawDialog(QWidget* parent = nullptr);
  ~DrawDialog();
  QImage* GetImage() { return image_; }

 protected:
  void paintEvent(QPaintEvent*) override;
  void mousePressEvent(QMouseEvent*) override;
  void mouseMoveEvent(QMouseEvent*) override;
  void mouseReleaseEvent(QMouseEvent*) override;

 private slots:
  void on_pushButtonClear_clicked();

 private:
  const int kSize = 512;
  const int kWidth = 70;
  Ui::DrawDialog* ui;
  QImage* image_;
  bool begin_draw_ = false;
  QPoint current_point_;
};

#endif  //  SRC_DRAWDIALOG_H_
