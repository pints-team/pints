function [qoi]=restingV(t,v)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

nv=length(v);

[mvax,imax]=max(v);
[vmin,imin]=min(v(imax:end));
qoi=vmin;

end

