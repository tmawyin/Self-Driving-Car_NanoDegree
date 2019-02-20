#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::cout;
using std::endl;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * TODO: Calculate the RMSE here.
   */
	VectorXd rmse(4);
	rmse << 0,0,0,0;

	// Checking that the estimation vector is not empty
	if (estimations.size() == 0)
	{
    	cout << "Error - Invalid size of the Estimation vector" << endl;
    	return rmse;
    }

    // Checking that the estimations and ground_truth vectors are of similar size
    if (estimations.size() != ground_truth.size())
	{
		cout << "Error - Invalid size of argument vectors" << endl;
		return rmse;
	}

	// Accumulate squared residuals
	for (int i=0; i < estimations.size(); ++i) {
		// Performing vector operation to calculate residuals
		VectorXd residual = estimations[i] - ground_truth[i];
		// Performing element-wise operation to find sum(residuals_squared)
		VectorXd residual_sqrd = residual.array()*residual.array();
		rmse += residual_sqrd;
	}

	// Calculating the mean
	rmse = rmse/estimations.size();

	// Calculating the squared root
	rmse = rmse.array().sqrt();

	// Returning the RMSE
	return rmse;

}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
   * TODO:
   * Calculate a Jacobian here.
   */

	// Initializing Jacobian (Hj) matrix variable
	MatrixXd Hj(3,4);

	// Recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	// Pre-compute a set of terms to avoid repeated calculation
	float c1 = px*px+py*py;
	float c2 = sqrt(c1);
	float c3 = (c1*c2);

	// Check division by zero
	if (fabs(c1) < 0.00001) 
	{
		cout << "CalculateJacobian () - Error - Division by Zero" << endl;
		return Hj;
	}

	// compute the Jacobian matrix
	Hj << (px/c2), (py/c2), 0, 0,
	  	 -(py/c1), (px/c1), 0, 0,
	  	  py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

	return Hj;
}
