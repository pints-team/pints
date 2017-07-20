function [nt,t,x,dxdp] = solve_map(x0,p,tfinal,xdim,kdim,solution_only)
     
if xdim ~= length(x0)
    fprintf('solve_map: Houston, we have a problem in x \n')
end
if kdim ~= length(p)
    fprintf('solve_map: Houston, we have a problem in p \n')
end

% Initialize arrays
% The final xdim sensitivities are wrt the initial conditions

nt=tfinal+1;
t=[0:tfinal];
x=zeros(xdim,tfinal+1);
dxdp=zeros(xdim,kdim+xdim,tfinal+1);
for i=1:xdim
    dxdp(i,kdim+i,1)=1;
end
    
x(:,1)=x0;
for it = 1:tfinal 
    x(:,it+1)=gvec(x(:,it),p);
end

if(solution_only == 0)    
    for it = 1:tfinal 
        dxdp(:,:,it+1)=drhs_dparam(x(:,it),dxdp(:,:,it),p);
    end
end

end
