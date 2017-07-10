#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

// TODO: Set the timestep length and duration
size_t N = 10;    // From 'Mind The Line', 25
double dt = .1;  // From 'Mind The Line', .05

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

// References
double ref_cte = 0;   // Reference Cross Track Error
double ref_epsi = 0;  // Reference orientation error
double ref_v = 40;    // Reference velocity

// Indices for vector of state variables and actuator variables
size_t x_start      = 0;
size_t y_start      = x_start + N;
size_t psi_start    = y_start + N;
size_t v_start      = psi_start + N;
size_t cte_start    = v_start + N;
size_t epsi_start   = cte_start + N;
size_t delta_start  = epsi_start + N;
size_t a_start      = delta_start + N - 1;

class FG_eval {
public:
    // Fitted polynomial coefficients
    Eigen::VectorXd coeffs;
    FG_eval(Eigen::VectorXd coeffs) { this->coeffs = coeffs; }

    typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
    void operator()(ADvector& fg, const ADvector& vars) {
      // TODO: implement MPC
      // `fg` a vector of the cost constraints, `vars` is a vector of variable values (state & actuators)
      // NOTE: You'll probably go back and forth between this function and
      // the Solver function below.
      fg[0] = 0;    // Cost, any cost should be added here

      // Attention factors
      const int att_cte = 500;
      const int att_epsi = 500;
      const int att_v = 1;
      const int att_delta = 5;
      const int att_a = 5;
      const int att_delta_diff = 50;
      const int att_a_diff = 25;

      // Cost w/ respect to reference states
      for (int t=0; t<N; t++) {
        fg[0] += att_cte  * CppAD::pow(vars[cte_start + t] - ref_cte, 2);    // cte -> 0
        fg[0] += att_epsi * CppAD::pow(vars[epsi_start + t] - ref_epsi, 2);  // epsi -> 0
        fg[0] += att_v    * CppAD::pow(vars[v_start + t] - ref_v, 2);        // velocity -> reference
      }

      // Cost w/ respect to minimize the use of actuators
      for (int t=0; t<N-1; t++) {
        fg[0] += att_delta  * CppAD::pow(vars[delta_start + t], 2);
        fg[0] += att_a      * CppAD::pow(vars[a_start + t], 2);
      }

      // Cost w/ respect to minimize the value gap between sequential actuations
      for (int t=0; t<N-2; t++) {
        fg[0] += att_delta_diff * CppAD::pow(vars[delta_start + t + 1] - vars[delta_start + t], 2);
        fg[0] += att_a_diff     * CppAD::pow(vars[a_start + t + 1] - vars[a_start + t], 2);
      }

      // Initial constraints
      fg[1 + x_start]     = vars[x_start];
      fg[1 + y_start]     = vars[y_start];
      fg[1 + psi_start]   = vars[psi_start];
      fg[1 + v_start]     = vars[v_start];
      fg[1 + cte_start]   = vars[cte_start];
      fg[1 + epsi_start]  = vars[epsi_start];

      // Rest of the constraints
      for (int t = 0; t<N-1; t++) {
        // State at time t+1
        AD<double> x1     = vars[x_start + t + 1];
        AD<double> y1     = vars[y_start + t + 1];
        AD<double> psi1   = vars[psi_start + t + 1];
        AD<double> v1     = vars[v_start + t + 1];
        AD<double> cte1   = vars[cte_start + t + 1];
        AD<double> epsi1  = vars[epsi_start + t + 1];
        // State at time t
        AD<double> x0     = vars[x_start + t];
        AD<double> y0     = vars[y_start + t];
        AD<double> psi0   = vars[psi_start + t];
        AD<double> v0     = vars[v_start + t];
        AD<double> cte0   = vars[cte_start + t];
        AD<double> epsi0  = vars[epsi_start + t];

        // Actuation at time t
        AD<double> delta0   = vars[delta_start + t];
        AD<double> a0       = vars[a_start + t];
        // Third order polynominal
        AD<double> f0       = coeffs[0] + coeffs[1] * x0 + coeffs[2] * x0 * x0 + coeffs[3] * x0 * x0 * x0;
        AD<double> psides0  = CppAD::atan(coeffs[1] + 2 * coeffs[2] * x0 + 3 * coeffs[3] * x0 * x0);

        // Model Constraints
        fg[2 + x_start + t] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
        fg[2 + y_start + t] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
        fg[2 + psi_start + t] = psi1 - (psi0 + v0 * delta0 / Lf * dt);
        fg[2 + v_start + t] = v1 - (v0 + a0 * dt);
        fg[2 + cte_start + t] = cte1 - ((f0 - y0) + (v0 * CppAD::sin(epsi0) * dt));
        fg[2 + epsi_start + t]  = epsi1 - ((psi0 - psides0) + v0 * delta0 / Lf * dt);

      }
    }
};

//
// MPC class definition implementation.
//
MPC::MPC() {}
MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
  bool ok = true;
  typedef CPPAD_TESTVECTOR(double) Dvector;

  // Extract values from state vector (for better readability)
  double x    = state[0];  // Position X
  double y    = state[1];  // Position Y
  double psi  = state[2];  // Orientation
  double v    = state[3];  // Velocity
  double cte  = state[4];  // Cross Track Error
  double epsi = state[5];  // Orientation Error

  // Set the number of model variables (includes both states and inputs).
  // N * state_dimension + (N-1) * actuators_dimension
  size_t n_vars = N * 6 + (N - 1) * 2;
  // Number of constraints
  size_t n_constraints = N * 6;

  // Initial value of the independent variables.
  // SHOULD BE 0 besides initial state.
  Dvector vars(n_vars);
  for (int i = 0; i < n_vars; i++) {
    vars[i] = 0;
  }

  // Set lower and upper limits for variables.
  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);

  // All non-actuator limits to minimum and maximum values
  for (int i=0; i<delta_start; i++) {
    vars_lowerbound[i] = -1.0e19;
    vars_upperbound[i] = 1.0e19;
  }

  // Delta limits to -25 and 25 degrees (values in radians)
  for (int i = delta_start; i<a_start; i++) {
    vars_lowerbound[i] = -0.436332 * Lf;
    vars_upperbound[i] = 0.436332 * Lf;
  }

  // Acceleration (positive and negative) limits
  for (int i=a_start; i<n_vars; i++) {
    vars_lowerbound[i] = -1.0;
    vars_upperbound[i] = 1.0;
  }

  // Lower and upper limits for the constraints
  // Should be 0 besides initial state.
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (int i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }

  constraints_lowerbound[x_start]     = x;
  constraints_lowerbound[y_start]     = y;
  constraints_lowerbound[psi_start]   = psi;
  constraints_lowerbound[v_start]     = v;
  constraints_lowerbound[cte_start]   = cte;
  constraints_lowerbound[epsi_start]  = epsi;

  constraints_upperbound[x_start]     = x;
  constraints_upperbound[y_start]     = y;
  constraints_upperbound[psi_start]   = psi;
  constraints_upperbound[v_start]     = v;
  constraints_upperbound[cte_start]   = cte;
  constraints_upperbound[epsi_start]  = epsi;

  // object that computes objective and constraints
  FG_eval fg_eval(coeffs);

  //
  // NOTE: You don't have to worry about these options
  //
  // options for IPOPT solver
  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  options += "Numeric max_cpu_time          0.5\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
          options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
          constraints_upperbound, fg_eval, solution);
  // Check some of the solution values
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  // Cost
  auto cost = solution.obj_value;
  std::cout << "Cost " << cost << std::endl;

  // TODO: Return the first actuator values. The variables can be accessed with
  // `solution.x[i]`.
  //
  // {...} is shorthand for creating a vector, so auto x1 = {1.0,2.0}
  // creates a 2 element double vector.

  vector<double> result;
  result.push_back(solution.x[delta_start]);
  result.push_back(solution.x[a_start]);

  for (int i=0; i<N-1; i++) {
    result.push_back(solution.x[x_start + i + 1]);
    result.push_back(solution.x[y_start + i + 1]);
  }
  return result;
}