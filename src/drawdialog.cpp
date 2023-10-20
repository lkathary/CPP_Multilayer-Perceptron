#include "drawdialog.h"

#include "ui_drawdialog.h"

DrawDialog::DrawDialog(QWidget *parent)
    : QDialog(parent),
      ui(new Ui::DrawDialog),
      image_(new QImage(512, 512, QImage::Format_RGB16)) {
  ui->setupUi(this);
  image_->fill(Qt::white);
}

DrawDialog::~DrawDialog() {
  delete ui;
  delete image_;
}

void DrawDialog::paintEvent(QPaintEvent *) {
  QPainter painter(this);
  painter.drawImage(QPoint(0, 0), *image_);

  painter.setPen(QPen(Qt::gray, 1));
  for (int i = 3; i < kSize; i += kSize / 28) {
    painter.drawLine(i, 0, i, kSize);
    painter.drawLine(0, i, kSize, i);
  }
}

void DrawDialog::mousePressEvent(QMouseEvent *event) {
  begin_draw_ = true;
  current_point_ = event->pos();
}

void DrawDialog::mouseMoveEvent(QMouseEvent *event) {
  if (begin_draw_) {
    QPainter painter(image_);
    painter.setPen(QPen(Qt::black, kWidth, Qt::SolidLine, Qt::RoundCap));
    QPoint point = event->pos();
    painter.drawLine(current_point_.x(), current_point_.y(), point.x(),
                     point.y());
    current_point_ = point;
    update();
  }
}

void DrawDialog::mouseReleaseEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    begin_draw_ = false;
  }
}

void DrawDialog::on_pushButtonClear_clicked() {
  image_->fill(Qt::white);
  update();
}
