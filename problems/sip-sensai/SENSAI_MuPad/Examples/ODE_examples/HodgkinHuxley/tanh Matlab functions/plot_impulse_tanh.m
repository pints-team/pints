function plot_impulse_tanh(a,b,k,r,m)
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
y=0.5*(tanh(k*(x-a)) - tanh(k*(x-b)));
plot(x,y)

end

