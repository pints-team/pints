function impulse_tanh(st1,st2,sp,sd,amp,k,r,m)
%
%  Construct a series of impulse functions , 
%
%   f = max(a,fn)
%
% using a tanh function
%
% a is the "floor"
% k is the "steepness" of the tanh function
% r is the range, i.e., x \in [-r,r]
% m is the number of points at which the function is plotted
%
% Example
%   max_fn_tanh(0.1,10000,1.5,500);
%

x=linspace(0,r,m);
fn=zeros(1,m);
y=zeros(1,m);
z=-ones(1,m);

for i=1:m
    if (x(i)>st1) & (x(i)<st2)
        y(i)=(x(i)-st1) - floor((x(i)-st1)/sp)*sp;
        z(i)=sd-y(i);
    end
end

stpx= tanh(k*z);
sleft= -0.5*(-1+stpx);
sright= 0.5*( 1+stpx);
fleft= 0;
fright= amp;
ftotal = sleft.*fleft + sright.*fright;
fn = ftotal;

figure(1)
subplot(1,3,1), plot(x,y,'or')
title('Function exceeding floor')
xlabel('x')
ylabel('y')

figure(1)
subplot(1,3,2), plot(x,z,'og')
title('Function exceeding floor')
xlabel('x')
ylabel('z')

subplot(1,3,3), plot(x,fn,'xk')
title('Floor')
xlabel('x')
ylabel('fn')
% 
% figure(2)
% plot(x,fn+gn,'-xg')
% title('max(fn,a)')
% xlabel('x')
% ylabel('(fn+gn)')

end

