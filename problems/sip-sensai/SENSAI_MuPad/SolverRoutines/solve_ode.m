function [nt,t,x,dxdp] = solve_ode(tt,x0,p,xdim,kdim,stype)
%
%  ***   [nt,t,x,dxdp] = solve_ode(tt,x0,p,xdim,kdim,stype)   ***
%
% If stype = 1, solution only
%          = 2, solution + sensitivity wrt parameters
%          = 3, solution + sensitivity wrt parameters + sensitivity wrt ics 
%

iprint=0;
if xdim ~= length(x0)
    fprintf('solve_ode: Houston, we have a problem in x \n')
end
if kdim ~= length(p)
    fprintf('solve_ode: Houston, we have a problem in p \n')
end

% Set initial conditions for solution and for sensitivity computations
% If stype = 1, then want only x0

if stype == 1
    xx0=x0;
    sdim=xdim;
    ncol=0;
elseif stype == 2
    xx0=[x0 zeros(xdim,kdim)];
    sdim=xdim*(1+kdim);
    ncol=kdim;
elseif stype == 3
    xx0=[x0 zeros(xdim,kdim) eye(xdim)];
    sdim=xdim*(1+kdim+xdim);
    ncol=xdim+kdim;
end

y0=reshape(xx0,sdim,1);

% Matlab routine ode45 solves vector systems of odes not matrix-valued odes
% Reshape the sensitivity computations to a vector systems by unpacking column by column
% to create a vector of length xdim*(1+kdim+xdim)

rtol=1E-6;
atol=1E-6;
options = odeset('RelTol',rtol,'AbsTol', atol*ones(sdim,1), 'Stats', 'off');


% Solve system of odes,
%      xdim nonlinear odes
%      kdim*xdim odes for sensitivity to parameters
%      xdim*xdim odes for sensitivity to initial conditions

% ode45 expects a handle to a function with two parameters t and y to create the rhs
% We therefore use a "nested function" gsystem, which expects five parameters,
%       t, y, p, xdim and kdim
%  to create the rhs
%  Here y is the column vector of length xdim*(1+kdim+xdim) containing the solution
%  and sensitivity information, i.e.,
%       y = [x; dxdp(:,1); dxdp(:,2); ... dxdp(:,kdim); ... dxdp(:,kdim+xdim)];

% fprintf('Solver: ode15s, Relative tolerance = %6.2e, Absolute tolerance = %6.2e \n',...
%          rtol, atol)
timerODE = tic;
[t,y]=ode15s(@(t,y) gsystem(t,y,p,xdim,kdim,ncol), tt, y0, options);
elapsed_timerODE = toc(timerODE);

if iprint >= 2
    fprintf('ODE solution time = %8.2f seconds \n', elapsed_timerODE)
end

% Reverse row and column order of y as returned by ode45
t=t';
y=y';
% fprintf('size(t) = (%5i, %5i) \n', size(t))
% fprintf('size(y) = (%5i, %5i) \n', size(y))
[ny,nt]=size(y);

% Map solution and sensitivities for printing and plotting
x(1:xdim,1:nt)=y(1:xdim,1:nt);

dxdp=zeros(1,1,1);
if stype >= 2 
    dxdp=zeros(xdim,ncol,nt);
    for k=1:ncol
        dxdp(1:xdim,k,1:nt)=y(k*xdim+1:(k+1)*xdim,1:nt);
    end
end




end

