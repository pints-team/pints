function step_fn_tanh(a,k,r,m)
%
%  Construct a step function at x=a 
%
%   f = fn(x),  x<a
%     = gn(x),  x>a
%
% using a tanh function
%
% a is the location of the discontinuity
% k is the "steepness" of the tanh function
% r is the range, i.e., x \in [-r,r]
% m is the number of points at which the function is plotted
%
% Example
%   step_fn_tanh(1,10000,pi,500);
%

x=linspace(-r,r,m);
stpx= tanh(k*(x-a)); 
sleft= -0.5*(-1+stpx); 
sright= 0.5*( 1+stpx); 
fleft= fn(x); 
fright= gn(x);
ftotal = sleft.*fleft + sright.*fright;

figure(1)
subplot(1,2,1),plot(x,fleft,'+r')
title('fn')
xlabel('x')
ylabel('f')

subplot(1,2,2),plot(x,sleft.*fleft,'+r')
title('Weighted fn')
xlabel('x')
ylabel('f')

figure(2)
subplot(1,2,1),plot(x,fright,'xk')
title('gn')
xlabel('x')
ylabel('g')

subplot(1,2,2),plot(x,sright.*fright,'xk')
title('Weigthed gn')
xlabel('x')
ylabel('g')

figure(3)
plot(x,ftotal,'-xg')
title('total')
xlabel('x')
ylabel('(f+g)')

end

function f = fn(x)
    f=2*cos(x);     
end

function g = gn(x)
    g=sin(x); 
end


