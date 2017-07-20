function [dgdx] = dgvec_dxvec(x,p) 

xdim=length(x);
dgdx = zeros(xdim,xdim); 

  dg1dx1 = -1.0/p(5)-(p(6)*x(3))/p(1);
  dg1dx2 = 0.0;
  dg1dx3 = -(p(6)*x(1))/p(1);
  dg1dx4 = 1.0/p(2);
  dg2dx1 = (p(6)*x(3))/p(1);
  dg2dx2 = -1.0/p(4)-1.0/p(5);
  dg2dx3 = (p(6)*x(1))/p(1);
  dg2dx4 = 0.0;
  dg3dx1 = 0.0;
  dg3dx2 = 1.0/p(4);
  dg3dx3 = -1.0/p(3)-1.0/p(5);
  dg3dx4 = 0.0;
  dg4dx1 = 0.0;
  dg4dx2 = 0.0;
  dg4dx3 = 1.0/p(3);
  dg4dx4 = -1.0/p(2)-1.0/p(5);

dgdx(1,1) = dg1dx1;
dgdx(1,2) = dg1dx2;
dgdx(1,3) = dg1dx3;
dgdx(1,4) = dg1dx4;
dgdx(2,1) = dg2dx1;
dgdx(2,2) = dg2dx2;
dgdx(2,3) = dg2dx3;
dgdx(2,4) = dg2dx4;
dgdx(3,1) = dg3dx1;
dgdx(3,2) = dg3dx2;
dgdx(3,3) = dg3dx3;
dgdx(3,4) = dg3dx4;
dgdx(4,1) = dg4dx1;
dgdx(4,2) = dg4dx2;
dgdx(4,3) = dg4dx3;
dgdx(4,4) = dg4dx4;

end