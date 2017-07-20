function [dgdp] = dgvec_dparam(t,x,p) 

xdim=length(x);
kdim=length(p);
dgdp = zeros(xdim,kdim+xdim); 

dg1dp1 = -(x(1)-4.0e1)*x(2)^3*x(3);
dg1dp2 = -(x(1)+8.7e1)*x(4)^4;
dg1dp3 = -x(1)-6.4387e1;
dg2dp1 = 0.0;
dg2dp2 = 0.0;
dg2dp3 = 0.0;
dg3dp1 = 0.0;
dg3dp2 = 0.0;
dg3dp3 = 0.0;
dg4dp1 = 0.0;
dg4dp2 = 0.0;
dg4dp3 = 0.0;

dgdp(1,1) = dg1dp1;
dgdp(1,2) = dg1dp2;
dgdp(1,3) = dg1dp3;
dgdp(2,1) = dg2dp1;
dgdp(2,2) = dg2dp2;
dgdp(2,3) = dg2dp3;
dgdp(3,1) = dg3dp1;
dgdp(3,2) = dg3dp2;
dgdp(3,3) = dg3dp3;
dgdp(4,1) = dg4dp1;
dgdp(4,2) = dg4dp2;
dgdp(4,3) = dg4dp3;

end