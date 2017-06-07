functions{
  real [] deriv_aslanidi(real t, real y, real[] theta, real[] x_r, real[] x_i){
  
  return dIdt;
  }
}

data{
  int N;
  real V[N];
  real I(N);
  real ts[N];
  real t0;
}

transformed data {
  real x_r[0];
  int x_i[0];
}


parameters{
  real<lower=0> p1;
  real<lower=0> p2;
  real<lower=0> p3;
  real<lower=0> p4;
  real<lower=0> p5;
  real p6;
  real<lower=0> p7;
  real<lower=0> p8;
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
  vector[N] I_int;
  I_int = integrate_ode_bdf(deriv_aslanidi, X0, t0, ts, theta, x_r, x_i);
  
  \\ likelihood
  for(i in 1:N){
    I[i] ~ normal(I_int[i],sigma);
  }
  
  \\priors
  p1 ~ normal(900,500);
  
  \\target += normal_lpdf(p1|900,200);
  
  \\target += normal_lpdf(I[i]|I_int[i],sigma);
}
