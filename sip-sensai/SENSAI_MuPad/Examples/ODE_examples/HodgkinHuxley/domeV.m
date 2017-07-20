function [qoi]=domeV(t,v)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

nv=length(v);

[vmax,imax]=max(v);
vv=v(1,imax:nv);
tt=t(imax:nv);

nvv=length(vv);
dvdt=zeros(1,nvv-1);

for i=1:nvv-1
    dvdt(i)=(vv(i+1)-vv(i))/(tt(i+1)-tt(i));
end

% figure(200)
% subplot(1,5,1),plot(t,v,'o')
% subplot(1,5,2),plot(tt,vv,'+')
% subplot(1,5,3),plot(tt(1:nvv-1),dvdt,'o')

[dvdtmin,imin]=min(dvdt);

ii=imin;
while dvdt(ii-1)>dvdt(ii)
    ii=ii-1;
end

% subplot(1,5,4),plot(tt(1:imin),dvdt(1:imin),'+')
% subplot(1,5,5),plot(tt(1:imin),vv(1:imin),'o')
domeV=vv(ii);
plateauV=tt(ii)-tt(1);

   
qoi=domeV;

end

