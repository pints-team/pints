function [qvals,qdim]=Input_checks(Qindex,Qvals,nq,fmin,fmax,fnedge,otype,ofname,omean,osigma,pnedge)

qdim=length(Qindex);
fprintf('Qindex \n')
disp(Qindex')

qvals=zeros(nq,qdim);
% qvals(1:nq,1:qdim)=Qvals(1:nq,Qindex(1:qdim));

% Compute functionals of Quantities of interest here
%
for idim=1:qdim
    if Qindex(idim)==6
        for iq=1:nq
            qvals(iq,idim)=Qfunc(Qvals(iq,:));
        end
    else   
        qvals(1:nq,idim)=Qvals(1:nq,Qindex(idim));
    end
end
%
%

fprintf('fmin \n')
disp(fmin')
fprintf('fmax \n', fmax)
disp(fmax')

fprintf('Minimum values of forward simulations \n')
sim_min=min(qvals,[],1);
disp(sim_min)
fprintf('Maximum values of forward simulations \n')
sim_max=max(qvals,[],1);
disp(sim_max)

for idim=1:qdim
    if fmin(idim)>sim_min(idim)
        fprintf('fmin(%3i) is greater than sim_min(%3i)... abort \n', idim, idim)
        return
    end
    if fmax(idim)<sim_max(idim)
        fprintf('fmax(%3i) is less than sim_max(%3i)... abort \n', idim, idim)
        return
    end
end

fprintf('fnedge \n')
disp(fnedge')
fprintf('otype \n')
disp(otype')
fprintf('Output distribution in file \n')
fprintf([ofname, '\n'])
fprintf('omean \n')
disp(omean')
fprintf('osigma \n')
disp(osigma')
fprintf('pnedge \n')
disp(pnedge')

end
