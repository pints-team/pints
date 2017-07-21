function solve_active_subspace_approximation(output_directory,nt,t,x,p,xdim,Mdim,Sdim,Fdim,Fp,U,S,V)
%
%  ***   solve_active_subspace_approximation(output_directory,nt,t,x,p,xdim,Mdim,Sdim,Fdim,Fp,U,S,V)   ***
%
%
%  Purpose
%  -------
%    Approximates using the active subspace
%
%  Variables
%  ---------
%
%
%  Calls
%  -----
%
%

%
global NCOLUMNS

np=11;


if Fdim==2
    for iF=1:Fdim
        deltap(iF,:)=linspace(-0.1,0.1,np)*p(iF);
    end    
      
    for it=2:nt
        UU(1:xdim,1:xdim)=U(1:xdim,1:xdim,it);
        SS(1:xdim,1:Fdim)=S(1:xdim,1:Fdim,it);
        VV(1:Fdim,1:Fdim)=V(1:Fdim,1:Fdim,it);
        
        Mapprox=zeros(xdim,Fdim)
        for is=1:Sdim
            Mapprox=Mapprox+SS(is,is)*UU(1:xdim,is)*VV(1:Fdim,is)';
        end
        
        xx=zeros(xdim,np,np);
        for i=1:np
            for j=1:np
                dp=[deltap(1,i); deltap(2,j)];
                xtemp=x(1:xdim,it)+Mapprox*dp;
                xx(1:xdim,i,j)=xtemp;
            end
        end
        xstore(1:xdim,1:np,1:np,it)=xx;
        
    end
    
end

ifig=15200;
for it=2:nt
    ifig=ifig+1;
    figure(ifig)
    yy(1:np,1:np)=xstore(1,1:np,1:np,it);
    p1=p(1)+deltap(1,1:np);
    p2=p(2)+deltap(2,1:np);
    mesh(p1,p2,yy')
    title(['Function approximation at t = ',num2str(t(it))])
    xlabel('p_1')
    ylabel('p_2')
    zlabel('x(t,p_1,p_2)')
end



end

