function min_fn_tanh( a, k, m)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% a is the minimum value permitted
% k is the steepness of the tanh function
% m is the number of points at which the function is plotted

a=0.1;
k=10000;
m=2000;
x=linspace(-2,2,m)

f = fn.*(0.5* (tanh(k*(fn-a))+1))
g = -a*ones(1,m).*(0.5*(tanh(k*(fn-a))-1));

figure(1)
subplot(1,2,1),plot(x,f,'+r')
subplot(1,2,2),plot(x,g,'xk')
figure(2)
plot(x,f+g,'-xg')


end


function [fn] = myfunction(x)
fn = (x.^3-x)
end
