<?
#
# cell.stan :: This will become the stan model definition file
#
# This file is part of Myokit
#  Copyright 2011-2017 Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Authors:
#  Ben Lambert
#  Michael Clerx
#
import myokit

?>/**
 * Automatically generated model from: <?=model.name()?>
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
<?
tab = '  '

print(2*tab)
print(2*tab + '// Parameters')
for k, var in enumerate(parameters):
    print(2*tab + 'real ' + v(var) + ' = parameters[' + str(k + 1) + '];')

print(2*tab)
print(2*tab + '// Constants')
for label, eq_list in equations.iteritems():
    eqs = []
    for eq in eq_list.equations(const=True, bound=False):
        if eq.lhs.var() not in parameters:
            eqs.append(eq)
    if eqs:
        print(2*tab + '// Component: ' + label)
        for eq in eqs:
            var = eq.lhs.var()
            if 'desc' in var.meta:
                print(2*tab + '// ' + '// '.join(
                    str(var.meta['desc']).splitlines()))
            print(2*tab + 'real ' + e(eq) + ';')

print(2*tab)
print(2*tab + '// States')    
for k, var in enumerate(model.states()):
    print(2*tab + 'real ' + v(var) + ' = state[' + str(k + 1) + '];')

print(2*tab)
print(2*tab + '// Calculate states')
for label, eq_list in equations.iteritems():
    eqs = []
    for eq in eq_list.equations(const=False, bound=False):
        if eq.lhs.var() not in parameters:
            eqs.append(eq)
    if eqs:
        print(2*tab + '// Component: ' + label)
        for eq in eqs:
            var = eq.lhs.var()
            if 'desc' in var.meta:
                print(2*tab + '// ' + '// '.join(
                    str(var.meta['desc']).splitlines()))
            print(2*tab + 'real ' + e(eq) + ';')

print(2*tab)
print(2*tab + '// Derivatives')
print(2*tab + 'real derivatives[' + str(model.count_states()) + '];')
for k, var in enumerate(model.states()):
    print(2*tab + 'derivatives[' + str(k + 1) + '] = ' + v(var.lhs()) + ';')

?>
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
  real<lower=0> p1;     // ms
  real<lower=0> p2;     // mV
  real<lower=0> p3;     // ms
  real<lower=0> p4;     // mV
  real<lower=0> p5;     // mV
  real p6;              // mV
  real<lower=0> p7;     // mV
  real<lower=0,upper=1> X0;
  real<lower=0> sigma;
}

transformed parameters{
  real theta[7];
  theta[1] = p1;
  theta[2] = p2;
  theta[3] = p3;
  theta[4] = p4;
  theta[5] = p5;
  theta[6] = p6;
  theta[7] = p7;
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
