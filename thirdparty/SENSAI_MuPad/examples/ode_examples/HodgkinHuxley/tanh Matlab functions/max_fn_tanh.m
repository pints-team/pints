function max_fn_tanh(a,k,r,m)
%
%  Construct a maximum function, 
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

x=linspace(-r,r,m);

fn = func(x).*  (0.5* ( tanh(k*(func(x)-a)) +1) );
gn = -a*ones(1,m)  .* (0.5* ( tanh(k*(func(x)-a)) -1) );

figure(1)
subplot(1,2,1), plot(x,fn,'+r')
title('Function exceeding floor')
xlabel('x')
ylabel('f')

subplot(1,2,2), plot(x,gn,'xk')
title('Floor')
xlabel('x')
ylabel('gn')

figure(2)
plot(x,fn+gn,'-xg')
title('max(fn,a)')
xlabel('x')
ylabel('(fn+gn)')

end


function f = func(x)
f = (x.^3-x);
end
