function [r0,dR0dp,R0warnings] = solve_R0(x,p,dxdp,kdim, NextGen) 

% This function solves r0 = r0(x,p) and its sensitivities at equilibrium!

[r0,dR0dx,dR0dparam,R0warnings] = r0_matrix(x,p);   % obtained from maple input

% I need to solve the system g(x,p) = 0 and dgdx*dxdp+dgdp = 0 simultaneously
% f = infected equations = gvec(NextGen)

% xold = 10*x;
% while norm(x-xold) > 10^-3
%     xold = x;
%     g = gvec(x,p);
%     dgdx = dgvec_dxvec(x,p);
%     
%     x = x - dgdx/g;
% 
% end
% dgdx = dgvec_dxvec(x,p);
% dgdp = dgvec_dparam(x,p);
% dgdp = dgdp(:,1:kdim);

% Only do this if dgdx is invertible
% dxdp = -dgdx\dgdp;     % Solve for dgdp at DFE -- can't do this for some
% models, so we require the user to use initial conditions of the DFE for
% R0 calculation

% Now solve dR0dp using chain rule
dR0dp=dR0dx*dxdp+dR0dparam;

end
