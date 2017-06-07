function [R0,dR0dx,dR0dp,R0warnings] = r0_matrix(x,p) 

warning0 = 0;

R0 = 0.0;

dR0dx1 = 0.0;
dR0dx2 = 0.0;
dR0dx3 = 0.0;
dR0dx4 = 0.0;
dR0dp1 = 0.0;
dR0dp2 = 0.0;
dR0dp3 = 0.0;

dR0dx(1) = dR0dx1;
dR0dx(2) = dR0dx2;
dR0dx(3) = dR0dx3;
dR0dx(4) = dR0dx4;
dR0dp(1) = dR0dp1;
dR0dp(2) = dR0dp2;
dR0dp(3) = dR0dp3;

warning1 = 0.0;
warning2_F = 1.0;
warning2_V = 0.0;
warning3 = 0.0;
warning4 = 1.0e2;
warning5 = 2.0;


R0warnings.w0 = warning0;
R0warnings.w1 = warning1;
R0warnings.w2F = warning2_F;
R0warnings.w2V = warning2_V;
R0warnings.w3 = warning3;
R0warnings.w4 = warning4;
R0warnings.w5 = warning5;

end