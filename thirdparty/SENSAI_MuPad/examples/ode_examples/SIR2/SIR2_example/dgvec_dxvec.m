function [dgdx] = dgvec_dxvec(t,x,p) 

xdim=length(x);
dgdx = zeros(xdim,xdim); 

  dg1dx1 = -p(1)*x(2);
  dg1dx2 = -p(1)*x(1)+p(2)+p(4);
  dg1dx3 = p(2);
  dg2dx1 = p(1)*x(2);
  dg2dx2 = p(1)*x(1)-p(2)-p(3)-p(4);
  dg2dx3 = 0.0;
  dg3dx1 = 0.0;
  dg3dx2 = p(3);
  dg3dx3 = -p(2);

dgdx(1,1) = dg1dx1;
dgdx(1,2) = dg1dx2;
dgdx(1,3) = dg1dx3;
dgdx(2,1) = dg2dx1;
dgdx(2,2) = dg2dx2;
dgdx(2,3) = dg2dx3;
dgdx(3,1) = dg3dx1;
dgdx(3,2) = dg3dx2;
dgdx(3,3) = dg3dx3;

end