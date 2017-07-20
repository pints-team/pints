function [dgdp] = dgvec_dparam(t,x,p) 

xdim=length(x);
kdim=length(p);
dgdp = zeros(xdim,kdim+xdim); 

  dg1dp1 = -((x(1)+x(2)+x(3))/p(2)-1.0)*(x(1)+x(2)+x(3));
  dg1dp2 = p(1)*1.0/p(2)^2*(x(1)+x(2)+x(3))^2;
  dg1dp3 = -x(1)*x(2);
  dg1dp4 = -x(1);
  dg1dp5 = 0.0;
  dg1dp6 = 0.0;
  dg2dp1 = 0.0;
  dg2dp2 = 0.0;
  dg2dp3 = x(1)*x(2);
  dg2dp4 = -x(2);
  dg2dp5 = -x(2);
  dg2dp6 = -x(2);
  dg3dp1 = 0.0;
  dg3dp2 = 0.0;
  dg3dp3 = 0.0;
  dg3dp4 = -x(3);
  dg3dp5 = x(2);
  dg3dp6 = 0.0;

dgdp(1,1) = dg1dp1;
dgdp(1,2) = dg1dp2;
dgdp(1,3) = dg1dp3;
dgdp(1,4) = dg1dp4;
dgdp(1,5) = dg1dp5;
dgdp(1,6) = dg1dp6;
dgdp(2,1) = dg2dp1;
dgdp(2,2) = dg2dp2;
dgdp(2,3) = dg2dp3;
dgdp(2,4) = dg2dp4;
dgdp(2,5) = dg2dp5;
dgdp(2,6) = dg2dp6;
dgdp(3,1) = dg3dp1;
dgdp(3,2) = dg3dp2;
dgdp(3,3) = dg3dp3;
dgdp(3,4) = dg3dp4;
dgdp(3,5) = dg3dp5;
dgdp(3,6) = dg3dp6;

end