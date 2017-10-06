/**
 * Automatically generated model from: Beattie-2017-IKr
 */
functions{

  /**
   * Uses bisection to find a value `x` in a vector `sorted`. 
   *
   * @param x The value to search for.
   * @param sorted A sorted (non-decreasing) vector of values to search in.
   * @return The greatest array indice `i` such that `sorted[i] <= x`
   */
  int find_interval_elem(real x, vector sorted) {
   
    int N;
    int iter;
    int max_iter;
    int left_ind;
    int right_ind;
    int mid_ind;
    real left;
    real right;
    real mid;
    
    N = num_elements(sorted);
    if(N == 0) return(0);

    left_ind  = 1;
    right_ind = N;

    max_iter = 100 * N;
    left  = sorted[left_ind ] - x;
    right = sorted[right_ind] - x;

    if(0 <= left)  return left_ind - 1;
    if(0 == right) return N - 1;
    if(0 >  right) return N;

    iter = 1;
    while((right_ind > left_ind + 1)  && (iter < max_iter)) {
      // is there a controlled way without being yelled at with a warning?
      mid_ind = (left_ind + right_ind) / 2;
      mid = sorted[mid_ind] - x;
      if (mid == 0) {
        return mid_ind - 1;
      }
      if (left  * mid < 0) {
        right_ind = mid_ind;
        right = mid;
      }
      if (right * mid < 0) {
        left_ind  = mid_ind;
        left  = mid;
      }
      iter = iter + 1;
    }
    if(iter == max_iter)
      print("Maximum number of iterations reached.");
    return left_ind;
  }
  
  /**
   * Calculates the value of the pace variable (i.e. the model input), based
   * on an externally defined time-series.
   */
  real get_pacing_value(real time, real[] xr, int[] xi) {

    // Split real inputs into times and values vector
    int n = xi[1];
    vector[n] times  = to_vector(xr[1:n]);
    vector[n] values = to_vector(xr[(n + 1):(2 * n)]);

    // Find indice for current time
    int i = find_interval_elem(time, values);
    
    // Return pacing value
    return (i == 0) ? values[1] : values[i];
  }
  
  /**
   * Calculates the model derivatives
   *
   *
   */
  real[] derivatives(real time, real[] state, real[] parameters, real[] xr,
      int[] xi) {
    
    // Get current pacing value
    real pace = get_pacing_value(time, xr, xi);
    
    // Parameters
    real p1 = parameters[1];
    real p2 = parameters[2];
    real p3 = parameters[3];
    real p4 = parameters[4];
    real p5 = parameters[5];
    real p6 = parameters[6];
    real p7 = parameters[7];
    real p8 = parameters[8];
    
    // Constants
    // Component: nernst
    real EK = -85.0;
    // Component: ikr
    real p9 = 0.1524;
    real g = p9;
    
    // States
    real open = state[1];
    real active = state[2];
    
    // Calculate states
    // Component: membrane
    // membane potential
    real V = pace;
    // Component: ikr
    real IKr = g * open * active * (V - EK);
    real k1 = p1 * exp(p2 * V);
    real k2 = p3 * exp(-p4 * V);
    real ikr_open_tau = 1.0 / (k1 + k2);
    real ikr_open_inf = k1 * ikr_open_tau;
    real d_open = (ikr_open_inf - open) / ikr_open_tau;
    real k4 = p7 * exp(-p8 * V);
    real k3 = p5 * exp(p6 * V);
    real ikr_active_tau = 1.0 / (k3 + k4);
    real ikr_active_inf = k4 * ikr_active_tau;
    real d_active = (ikr_active_inf - active) / ikr_active_tau;
    
    // Derivatives
    real derivatives[2];
    derivatives[1] = d_open;
    derivatives[2] = d_active;

    return derivatives;
  }
  
  vector solve_forced_ode(real[] ts, real X0, real[] theta, real[] V, real t0){
    int x_i[1];
    real I[size(V),1];
    x_i[1] = size(V);

    I = integrate_ode_bdf(derivatives, rep_array(X0, 1), t0, ts, theta, to_array_1d(append_row(to_vector(ts), to_vector(V))), x_i);
    return(to_vector(I[,1]));
  }
}

data{
  int N;
  real V[N];
  real I[N];
  real ts[N];
  real t0;
}

transformed data {
  int x_i[0];
}


parameters{
  real<lower=1e-7,upper=1> p1; // [1/ms]
  real<lower=1e-7,upper=1> p2; // [1/mV]
  real<lower=1e-7,upper=1> p3; // [1/ms]
  real<lower=1e-7,upper=1> p4; // [1/mV]
  real<lower=1e-7,upper=1> p5; // [1/ms]
  real<lower=1e-7,upper=1> p6; // [1/mV]
  real<lower=1e-7,upper=1> p7; // [1/ms]
  real<lower=1e-7,upper=1> p8; // [1/mV]
  real<lower=0,upper=1> X0;
  real<lower=0> sigma;
}

transformed parameters{
  real theta[8];
  theta[1] = p1;
  theta[2] = p2;
  theta[3] = p3;
  theta[4] = p4;
  theta[5] = p5;
  theta[6] = p6;
  theta[7] = p7;
  theta[8] = p8;

}

model{
  // solve ODE using stiff solver
  vector[N] I_int;
  I_int = solve_forced_ode(ts, X0, theta, V,-0.1);
  
  // likelihood
  for(i in 1:N){
    I[i] ~ normal(I_int[i],sigma);
  }
  
  //priors
  p1 ~ normal(900,500);
  p2 ~ normal(5,1);
  p3 ~ normal(100,10);
  p4 ~ normal(0.1,0.02);
  p5 ~ normal(12.25,3);
  p6 ~ normal(-5.6,1);
  p7 ~ normal(20.4,3);
  sigma ~ normal(1,0.1);
}
