function plot_projections2active_2D(pmin,pmax,psim,pp,pem,psample,...
                                    isim,ivol,i1,i2,nactive,ifig)
%
%  ***   plot_projections2active(pmin,pmax,psim,pp,pem,psample,...
%                                isim,ivol,i1,i2,nactive,ifig)  ***
%
% Plot projections on to active subspace and node assignments for samples
%

[kdim,nsim]=size(psim);
% fprintf('plot_projections2active \n')

figure(ifig)
axis([0.99*pmin(1) 1.01*pmax(1) 0.99*pmin(2) 1.01*pmax(2)])
axis square
title('Approximation of objective function at random samples')
xlabel('Parameter 1')
ylabel('Parameter 2')
hold on

xx=psample(1,ivol);
yy=psample(2,ivol);
plot(xx,yy,'sb')
for ii=1:nactive
    xx=psample(1,i1+ii-1);
    yy=psample(2,i1+ii-1);
    plot(xx,yy,'db')
end
plot([psample(1,i1),psample(1,i2)],[psample(2,i1),psample(2,i2)],'--b')

xproj=[psim(1,isim) pp(1)];
yproj=[psim(2,isim) pp(2)];
xem=[psim(1,isim) pem(1,isim)];
yem=[psim(2,isim) pem(2,isim)];

if mod(ivol,4)==1
    plot(xproj,yproj,'--b')
    plot(xem(1),yem(1),'om',xem(2),yem(2),'.m',xem,yem,'-m')
elseif mod(ivol,4)==2
    plot(xproj,yproj,'--b')
    plot(xem(1),yem(1),'og',xem(2),yem(2),'.g',xem,yem,'-g')
elseif mod(ivol,4)==3
    plot(xproj,yproj,'--b')
    plot(xem(1),yem(1),'oc',xem(2),yem(2),'.c',xem,yem,'-c')
else
    plot(xproj,yproj,'--b')
    plot(xem(1),yem(1),'ok',xem(2),yem(2),'.k',xem,yem,'-k')
end


end

