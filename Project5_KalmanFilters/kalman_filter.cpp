#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
   * TODO: predict the state
   */
  x_ = F_ * x_;
  P_ = F_*P_*F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
   * TODO: update the state by using Kalman Filter equations
   */
  VectorXd y_ = z - (H_*x_);
  MatrixXd S_ = H_*P_*H_.transpose() + R_;
  MatrixXd K_ = P_*H_.transpose()*S_.inverse();  
   
  // New state
  x_ = x_ + (K_ * y_);
  // Setting up the Identity matrix
  MatrixXd I = MatrixXd::Identity(x_.size(),x_.size());
  P_ = (I - K_*H_)*P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   * TODO: update the state by using Extended Kalman Filter equations
   */
  // Initializing required variables
  VectorXd h_x = VectorXd(3);

  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);

  // Calculating h(x'); i.e. mapping state vector to polar coordinates
  float rho = sqrt(px*px + py*py);
  float theta = atan2(py, px);
  float rho_dot;
  // Checking if we are dividing by zero
  if (fabs(rho) < 0.0001)
  {
    rho_dot = 0.0001;
  }else{
    rho_dot = (px*vx + py*vy)/rho;
  }

  h_x << rho, theta, rho_dot;
  
  VectorXd y_ = z - h_x;
  // Normalizing the angle
  while (y_(1) > M_PI)  y_(1) -= M_PI;
  while (y_(1) < -M_PI) y_(1) += M_PI;

  MatrixXd S_ = H_*P_*H_.transpose() + R_;
  MatrixXd K_ = P_*H_.transpose()*S_.inverse();  
   
  // New state
  x_ = x_ + (K_ * y_);
  // Setting up the Identity matrix
  MatrixXd I = MatrixXd::Identity(x_.size(),x_.size());
  P_ = (I - K_ *H_)*P_; 
}
