function [q,fmin,fmax,fedges,pout]=brute_force(x0,p0,tfinal,ntsteps,xdim,kdim,stype,rtype,nr,idim,nss,nedges);
%  ***   [q,fmin,fmax,fedges,pout]=brute_force(x0,p0,tfinal,ntsteps,xdim,kdim,stype,rtype,nr,idim,nss,nedges);
%
% x0 = initial conditions
% p0 = base values for parameters
% tfinal = time at end of simulation
% nststeps = number of time steps in the solution
%            = 0 -> allow solver to choose solution values
% xdim = number of variables
% kdim = number of parameters
% stype = 1 -> compute solutions only
%       = 2 -> compute solution and sensitivities wrt parameters
%       = 3 -> compute solution and sensitivities wrt parameters and initial conditions
% rtype = 0 -> choose parameters from a uniform distribution
%       = 1 -> choose parameters from a normal distribution
% nr    = number of realizations
% idim  = parameter to be chosen at random
% nss   = number of summary statistics
% nedge = number of bins in histograms of summary statistics

fmin=zeros(1,nss);
fmax=zeros(1,nss);
nedge=max(nedges);
fedges=zeros(nss,nedge);
pout=zeros(nss,nedge);

if ntsteps==0
    solntimes=[];
end
pert=1;
DIR='dummy';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Vary p(idim) keeping other parameters fixed
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if rtype==0
    pmin=0.5*p0(idim);
    pmax=2.0*p0(idim);
    pmean=(pmin+pmax)/2;
    prange=(pmax-pmin)/2;
    pp=pmean+prange*(-1+2*rand(1,nr));
elseif rtype==1
    pp=p0(idim)*(1+0.1*randn(1,nr));
end
p=repmat(p0,1,nr);
p(idim,1:nr)=pp(1:nr);

for ir=1:nr
    [yy,tt]=ode_setup(tfinal,ntsteps,solntimes,pert,DIR);
    [nt,t,x,dxdp]=solve_ode(tt,x0,p(1:kdim,ir),xdim,kdim,stype);
    v=x(1,:); 
    q(1,ir)=APD(t,v,0.5);
    q(2,ir)=APD(t,v,0.9);
    q(3,ir)=maxdVdt(t,v);
    q(4,ir)=maxV(t,v);
    q(5,ir)=restingV(t,v);
end

figure(idim)
for i=1:5
    subplot(2,3,i),hist(q(i,1:nr),nedges(i))
    title(['Varying p(',num2str(idim),')'])
    xlabel(['q(',num2str(i),')'])
    ylabel('Frequency')
    m(idim,i)=mean(q(i,1:nr));
    s(idim,i)=std(q(i,1:nr));
end

for i=1:5
    fmin(i)=min(q(i,1:nr));
    fmax(i)=max(q(i,1:nr));
    fmin(i)=fmin(i)-0.01*abs(fmin(i));
    fmax(i)=fmax(i)+0.01*abs(fmax(i));
    fedges(i,:)=linspace(fmin(i),fmax(i),nedges(i));
    [n,bin]=histc(q(i,1:nr),fedges(i,:));
    nn=sum(n);
    for j=1:nedges(i)
        pout(i,j)=n(j)/nn;
    end
end

fprintf('Output probabilities for p(%2i) \n',idim)
disp(pout)
sum(pout,2)


end

