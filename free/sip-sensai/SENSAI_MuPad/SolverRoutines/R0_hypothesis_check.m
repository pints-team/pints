function R0_hypothesis_check(r0,dR0dp,R0warnings)
% See if there is a condition in the theorem that does not hold!

if R0warnings.w0 == 1
    fprintf('R0 sensitivities are not available:\nMuPad is unable to calculate the analytical R0 from the Next Generation Matrix.\nPerhaps there are too many infected classes.\nFor efficiency, set R0_only = 1.\n');
    msgbox('WARNING: R0 sensitivities are not available.  MuPad is unable to calculate the analytical R0 from the Next Generation Matrix.  Perhaps there are too many infected classes.  For efficiency, set R0_only = 1.');
    % force R0_only = 0 for the remainder of run
    R0_only = 1;
end

if R0warnings.w1 == 1
    fprintf('R0 is NOT VALID:\nThe transition matrix is not asymptotically stable\n');
    msgbox('WARNING: R0 is NOT VALID.  The transition matrix is not asymptotically stable.');
end

if R0warnings.w2F == 1
    fprintf('R0 is NOT VALID:\nThe fecundity matrix F is not nonnegative\n');
    msgbox('WARNING: R0 is NOT VALID.  The fecundity matrix F is not nonnegative.');
end

if R0warnings.w2V == 1
    fprintf('R0 is NOT VALID:\nThe transition matrix T is not nonnegative\n');
    msgbox('WARNING: R0 is NOT VALID.  The transition matrix T is not nonnegative.');
end

if R0warnings.w3 == 1
    fprintf('R0 is NOT VALID:\nThe equilibrium is not asymptotically stable in the absence of disease\n');
    msgbox('WARNING: R0 is NOT VALID.  The equilibrium is not asymptotically stable in the absence of disease.');
end

if r0<0
    fprintf('R0 is NOT VALID:\nR0 must be a nonnegative number\n');
    msgbox('WARNING: R0 is NOT VALID.  R0 must be a nonnegative number.');
end

if R0warnings.w4 ~= 0
    if R0warnings.w4 == 100
        fprintf('R0 is NOT VALID:\nThe transition matrix V singular.\nPerhaps SENSAI does not correctly identify the placement of terms in F and V.\n');
        msgbox('WARNING: R0 is NOT VALID.  The transition matrix V singular.  Perhaps SENSAI does not correctly identify the placement of terms in F and V.');
    else
        fprintf('R0 is NOT VALID:\nEquation %d is identically 0 and makes the transition matrix V singular\n', R0warnings.w4);
        msgbox(['WARNING: R0 is NOT VALID.  Equation ' num2str(R0warnings.w4),' is identically 0 and makes the transition matrix V singular.']);
    end
end

if R0warnings.w5 ~= 0
    fprintf('R0 is NOT VALID:\nThe disease-free subspace is not invariant.  Disease can enter state %d even in the absence of infection\n',R0warnings.w5);
    msgbox(['WARNING: R0 is NOT VALID.  The disease-free subspace is not invariant.  Disease can enter state ',num2str(R0warnings.w5), ' even in the absence of infection.']);
end



end

