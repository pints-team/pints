function [dgdx] = dgvec_dxvec(t,x,p) 

xdim=length(x);
dgdx = zeros(xdim,xdim); 

  dg1dx1 = -p(2)-p(3);
  dg1dx2 = p(1);
  dg1dx3 = p(1);
  dg1dx4 = p(1);
  dg1dx5 = p(1);
  dg2dx1 = p(2);
  dg2dx2 = -p(4)*x(4)-p(5)-p(10);
  dg2dx3 = 0.0;
  dg2dx4 = -p(4)*x(2);
  dg2dx5 = p(9);
  dg3dx1 = 0.0;
  dg3dx2 = p(4)*x(4);
  dg3dx3 = -1.0/p(6)-p(5);
  dg3dx4 = p(4)*x(2);
  dg3dx5 = 0.0;
  dg4dx1 = 0.0;
  dg4dx2 = 0.0;
  dg4dx3 = 1.0/p(6);
  dg4dx4 = -1.0/p(8)-p(5)-p(7);
  dg4dx5 = 0.0;
  dg5dx1 = 0.0;
  dg5dx2 = p(10);
  dg5dx3 = 0.0;
  dg5dx4 = 0.0;
  dg5dx5 = -p(5)-p(9);

dgdx(1,1) = dg1dx1;
dgdx(1,2) = dg1dx2;
dgdx(1,3) = dg1dx3;
dgdx(1,4) = dg1dx4;
dgdx(1,5) = dg1dx5;
dgdx(2,1) = dg2dx1;
dgdx(2,2) = dg2dx2;
dgdx(2,3) = dg2dx3;
dgdx(2,4) = dg2dx4;
dgdx(2,5) = dg2dx5;
dgdx(3,1) = dg3dx1;
dgdx(3,2) = dg3dx2;
dgdx(3,3) = dg3dx3;
dgdx(3,4) = dg3dx4;
dgdx(3,5) = dg3dx5;
dgdx(4,1) = dg4dx1;
dgdx(4,2) = dg4dx2;
dgdx(4,3) = dg4dx3;
dgdx(4,4) = dg4dx4;
dgdx(4,5) = dg4dx5;
dgdx(5,1) = dg5dx1;
dgdx(5,2) = dg5dx2;
dgdx(5,3) = dg5dx3;
dgdx(5,4) = dg5dx4;
dgdx(5,5) = dg5dx5;

end