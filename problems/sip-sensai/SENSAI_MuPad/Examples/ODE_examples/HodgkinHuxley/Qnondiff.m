function qsample=Qnondiff(t,x,p,Qindex,nrsample,obserror)
%
%  ***   qsample=Qnondiff(t,x,p,Qindex,nrandom,obserror)   ***
%
% Select from a number of quantities of interest
%
% 1 -> APD50
% 2 -> APD90
% 3 -> max(dV/dt)
% 4 -> max(V)
% 5 -> resting potential = V(end)
%

nq=length(Qindex);
qsample=zeros(nq,nrsample);
[mx,nx]=size(x);
v=x(1,1:nx);

for ir=1:nrsample
    % vr=v.*(1+obserror*randn(1,nx));  % Relative error
    vr=v+obserror*randn(1,nx);         % Absolute error
    for iq=1:nq
        switch Qindex(iq)
            case 1
                [qsample(iq,ir),flag]=APD(t,vr,0.5);
                if flag ~=0
                    fprintf('APD50: flag=%2i, ir=%5i \n', flag, ir)
                    disp(p')
                end
            case  2
                [qsample(iq,ir),flag]=APD(t,vr,0.9);
                if flag ~=0
                    fprintf('APD90: flag=%2i, ir=%5i \n', flag, ir)
                    disp(p')
                end
            case 3
                qsample(iq,ir)=maxdVdt(t,vr);
            case 4
                qsample(iq,ir)=maxV(t,vr);
            case 5
                qsample(iq,ir)=restingV(t,vr);
            case 6
                qsample(iq,ir)=domeV(t,vr);
            case 7
                qsample(iq,ir)=plateauV(t,vr);
        end
    end
end

end

