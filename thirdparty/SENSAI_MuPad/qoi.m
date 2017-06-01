function [q,dqdx,dqdp] = qoi(t,x,p,xdim,kdim,qdim) 

q=zeros(qdim,1);
dqdx=zeros(qdim,xdim);
dqdp=zeros(qdim,kdim+xdim); 

q1 = x(1);

q(1) = q1;

dq1dx1 = 1.0;

dqdx(1,1) = dq1dx1;

dq1dp1 = 0.0;
dq1dp2 = 0.0;

dqdp(1,1) = dq1dp1;
dqdp(1,2) = dq1dp2;

end