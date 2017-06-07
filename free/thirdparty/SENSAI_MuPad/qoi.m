function [q,dqdx,dqdp] = qoi(t,x,p,xdim,kdim,qdim) 

q=zeros(qdim,1);
dqdx=zeros(qdim,xdim);
dqdp=zeros(qdim,kdim+xdim); 

q1 = x(1);

q(1) = q1;

dq1dx1 = 1.0;
dq1dx2 = 0.0;
dq1dx3 = 0.0;
dq1dx4 = 0.0;

dqdx(1,1) = dq1dx1;
dqdx(1,2) = dq1dx2;
dqdx(1,3) = dq1dx3;
dqdx(1,4) = dq1dx4;

dq1dp1 = 0.0;
dq1dp2 = 0.0;
dq1dp3 = 0.0;

dqdp(1,1) = dq1dp1;
dqdp(1,2) = dq1dp2;
dqdp(1,3) = dq1dp3;

end