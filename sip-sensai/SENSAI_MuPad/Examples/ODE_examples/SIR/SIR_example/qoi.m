function [q,dqdx,dqdp] = qoi(t,x,p,xdim,kdim,qdim) 

q=zeros(qdim,1);
dqdx=zeros(qdim,xdim);
dqdp=zeros(qdim,kdim+xdim); 

  q1 = x(2)/(x(1)+x(2)+x(3));
  q2 = x(2);

q(1) = q1;
q(2) = q2;

  dq1dx1 = -x(2)*1.0/(x(1)+x(2)+x(3))^2;
  dq1dx2 = 1.0/(x(1)+x(2)+x(3))-x(2)*1.0/(x(1)+x(2)+x(3))^2;
  dq1dx3 = -x(2)*1.0/(x(1)+x(2)+x(3))^2;
  dq2dx1 = 0.0;
  dq2dx2 = 1.0;
  dq2dx3 = 0.0;

dqdx(1,1) = dq1dx1;
dqdx(1,2) = dq1dx2;
dqdx(1,3) = dq1dx3;
dqdx(2,1) = dq2dx1;
dqdx(2,2) = dq2dx2;
dqdx(2,3) = dq2dx3;

  dq1dp1 = 0.0;
  dq1dp2 = 0.0;
  dq1dp3 = 0.0;
  dq1dp4 = 0.0;
  dq1dp5 = 0.0;
  dq1dp6 = 0.0;
  dq2dp1 = 0.0;
  dq2dp2 = 0.0;
  dq2dp3 = 0.0;
  dq2dp4 = 0.0;
  dq2dp5 = 0.0;
  dq2dp6 = 0.0;

dqdp(1,1) = dq1dp1;
dqdp(1,2) = dq1dp2;
dqdp(1,3) = dq1dp3;
dqdp(1,4) = dq1dp4;
dqdp(1,5) = dq1dp5;
dqdp(1,6) = dq1dp6;
dqdp(2,1) = dq2dp1;
dqdp(2,2) = dq2dp2;
dqdp(2,3) = dq2dp3;
dqdp(2,4) = dq2dp4;
dqdp(2,5) = dq2dp5;
dqdp(2,6) = dq2dp6;

end