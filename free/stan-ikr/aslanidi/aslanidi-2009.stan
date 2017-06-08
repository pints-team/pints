functions{
  int find_interval_elem(real x, vector sorted, int start_ind){
    int res;
    int N;
    int max_iter;
    real left;
    real right;
    int left_ind;
    int right_ind;
    int iter;

    N = num_elements(sorted);

    if(N == 0) return(0);

    left_ind  = start_ind;
    right_ind = N;

    max_iter = 100 * N;
    left  = sorted[left_ind ] - x;
    right = sorted[right_ind] - x;

    if(0 <= left)  return(left_ind-1);
    if(0 == right) return(N-1);
    if(0 >  right) return(N);

    iter = 1;
    while((right_ind - left_ind) > 1  && iter != max_iter) {
      int mid_ind;
      real mid;
      // is there a controlled way without being yelled at with a
      // warning?
      mid_ind = (left_ind + right_ind) / 2;
      mid = sorted[mid_ind] - x;
      if (mid == 0) return(mid_ind-1);
      if (left  * mid < 0) { right = mid; right_ind = mid_ind; }
      if (right * mid < 0) { left  = mid; left_ind  = mid_ind; }
      iter = iter + 1;
    }
    if(iter == max_iter)
      print("Maximum number of iterations reached.");
    return(left_ind);
  }
  
  real[] deriv_aslanidi(real t, real[] I, real[] theta, real[] x_r, int[] x_i){
    
    int aLen = x_i[1];
    vector[aLen] ts = to_vector(x_r[1:aLen]);
    vector[aLen] V = to_vector(x_r[(aLen+1):(2*aLen)]);
    int aT = find_interval_elem(t, ts, 1);
    real aV = (aT==0) ? V[1] : V[aT];
    
    real xtau = theta[1] / (1 + exp(aV/ theta[2])) + theta[3];
    real xinf = 1 / (1 + exp(-(aV + theta[4]) / theta[5]));
    real rinf = 1 / (1 + exp((aV + theta[6]) / theta[7]));
    real dydt[1];
    dydt[1] = (xinf - I[1]) / xtau;
    return dydt;
  }
  
  vector solve_aslanidi_forced_ode(real[] ts, real X0, real[] theta, real[] V, real t0){
    int x_i[1];
    real I[size(V),1];
    x_i[1] = size(V);

    I = integrate_ode_bdf(deriv_aslanidi, rep_array(X0, 1), t0, ts, theta, to_array_1d(append_row(to_vector(ts), to_vector(V))), x_i);
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
  I_int = solve_aslanidi_forced_ode(ts, X0, theta, V,-0.1);
  
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