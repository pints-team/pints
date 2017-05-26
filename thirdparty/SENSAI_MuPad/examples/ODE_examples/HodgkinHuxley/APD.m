function [qoi,flag]=APD(t,v,frac)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

flag=0;
qoi=0;
nv=length(v);

[vmax,imax]=max(v);
v1=v(1,1:imax-1);
v2=v(1,imax:nv);

[vmin,imin]=min(v2);
vcrit=vmin+(1-frac)*(vmax-vmin);

ipd1=find(v1-vcrit>0,1);
ipd2=find(v2-vcrit<0,1);

if ipd1 > 1
    pp1=(vcrit-v1(ipd1-1))/(v1(ipd1)-v1(ipd1-1));
    s1=t(ipd1-1);
    s2=t(ipd1);
    t1=s1+pp1*(s2-s1);
elseif ipd1 == 1
    flag=1;
    t1=0;    
end

if ipd2 > 1
    pp2=(vcrit-v2(ipd2-1))/(v2(ipd2)-v2(ipd2-1));
    s1=t(imax+ipd2-1);
    s2=t(imax+ipd2);
    t2=s1+pp2*(s2-s1);
elseif ipd2 == 1
    flag=2;
    t2=t(imax);
end

if isempty(ipd1) | isempty(ipd2)
    flag=3;
else
    qoi=t2-t1;
end

end

