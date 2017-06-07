functions{
  real deriv_aslanidi(real t, real y, real[] theta, real[] V, real[] x_i){

    real xtau = theta[1] / (1 + exp(V / theta[2])) + theta[3];
    real xinf = 1 / (1 + exp(-(V + theta[4]) / theta[5]));
    real rinf = 1 / (1 + exp((V + theta[6]) / theta[7]));
    //real EK = -85;
    //IKr = p8 * xr * rr * (V - EK);
    real dydt = (xinf - y) / xtau;
    return dydt;
  }
}

data{
  int N;
  real V[N,1];
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
  real<lower=0> p8;     // E, in mS/uF
  real<lower=0,upper=1> X0[1];
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
  vector[N] I_int;
  I_int = integrate_ode_bdf(deriv_aslanidi, X0, t0, ts, theta, V, x_i);
  
  // likelihood
  for(i in 1:N){
    I[i] ~ normal(I_int[i],sigma);
  }
  
  //priors
  p1 ~ normal(900,500);
  
  //target += normal_lpdf(p1|900,200);
  
  //target += normal_lpdf(I[i]|I_int[i],sigma);
}
