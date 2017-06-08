function plot_projections2active_3D(pmin,pmax,psim,pp,pem,psample,...
                                    isim,ivol,i1,i2,nactive,ifig)
%
%  ***   plot_projections2active(pmin,pmax,psim,pp,pem,psample,isim,ivol,i1,i2,nactive)  ***
%
% Plot projections on to active subspace and node assignments for samples
%

[kdim,nsim]=size(psim);
% fprintf('plot_projections2active_3D \n')

figure(ifig)
axis([0.99*pmin(1) 1.01*pmax(1) 0.99*pmin(2) 1.01*pmax(2) 0.99*pmin(3) 1.01*pmax(3)])
axis square
title('Approximation of objective function at random samples')
xlabel('Parameter 1')
ylabel('Parameter 2')
zlabel('Parameter 3')
hold on

xx=psample(1,ivol);
yy=psample(2,ivol);
zz=psample(3,ivol);
plot(xx,yy,'sb')
for ii=1:nactive
    xx=psample(1,i1+ii-1);
    yy=psample(2,i1+ii-1);
    zz=psample(3,i1+ii-1);
    plot3(xx,yy,zz,'db')
end
plot3([psample(1,i1),psample(1,i2)],[psample(2,i1),psample(2,i2)],[psample(3,i1),psample(3,i2)],'--b')

xproj=[psim(1,isim) pp(1)];
yproj=[psim(2,isim) pp(2)];
zproj=[psim(3,isim) pp(3)];
xem=[psim(1,isim) pem(1,isim)];
yem=[psim(2,isim) pem(2,isim)];
zem=[psim(3,isim) pem(3,isim)];

if mod(ivol,4)==1
    plot3(xproj,yproj,zproj,'--b')
    plot3(xem(1),yem(1),zem(1),'om',xem(2),yem(2),zem(2),'.m',xem,yem,zem,'-m')
elseif mod(ivol,4)==2
    plot3(xproj,yproj,zproj,'--b')
    plot3(xem(1),yem(1),zem(1),'og',xem(2),yem(2),zem(2),'.g',xem,yem,zem,'-g')
elseif mod(ivol,4)==3
    plot3(xproj,yproj,zproj,'--b')
    plot3(xem(1),yem(1),zem(1),'oc',xem(2),yem(2),zem(2),'.c',xem,yem,zem,'-c')
else
    plot3(xproj,yproj,zproj,'--b')
    plot3(xem(1),yem(1),zem(1),'ok',xem(2),yem(2),zem(2),'.k',xem,yem,zem,'-k')
end

index=[1 1 2; 2 3 3];
for islice=1:3
    ifigslice=ifig+islice;
    p1=index(1,islice);
    p2=index(2,islice);
    qmin=[pmin(p1);pmin(p2)];
    qmax=[pmax(p1);pmax(p2)];
    qsim=[psim(p1,:);psim(p2,:)];
    qq=[pp(p1);pp(p2)];
    qem=[pem(p1,:);pem(p2,:)];
    qsample=[psample(p1,:);psample(p2,:)];
    plot_projections2active_2D(qmin,qmax,qsim,qq,qem,qsample,...
                               isim,ivol,i1,i2,nactive,ifigslice)
end


end

