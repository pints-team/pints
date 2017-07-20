function [yy]=stretch_grid(xmin,xmax,n,r);
%  ***   [xx]=stretch_grid(xmin,xmax,n,r);   ***
%
% Creates a 1D mesh with 2n+1 points between xmin and xmax that is 
% exponentially stretched towards the middle

xmid=0.5*(xmin+xmax);
y=linspace(1,exp(r),n+1);
x=(log(y)/r);
xx(1:n)=xmin+(xmid-xmin)*x(1:n);
xx(n+1)=xmid;
xx(n+2:2*n+1)=xmid+(xmax-xmid)*(1-x(n:-1:1));
% xx
% figure(98)
% plot(1:2*n+1,xx,':o')


% Find midpoints
yy(1:n)=0.5*(xx(1:n)+xx(2:n+1));
yy(n+1:2*n)=0.5*(xx(n+1:2*n)+xx(n+2:2*n+1));
% yy
% figure(99)
% plot([1.5:2*n+0.5],yy,':s')
% pause

end

