function [qoi]=maxV(t,v)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

nv=length(v);
dvdt=zeros(1,nv-1);
for i=1:nv-1
    dvdt(i)=(v(i+1)-v(i))/(t(i+1)-t(i));
end
[qoi,imax]=max(dvdt);

end

