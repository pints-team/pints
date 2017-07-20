function SIP_datafiles(output_directory,qdim,kdim, ...
                       pmin,pmax,qmin,qmax, ...
                       nobs,nvol,vol,nrsample,psample,qsample, ...
                       pref,qref,cvar,nedge)


% Create binary file for SIP
Ldim=kdim;
Lmin=pmin';
Lmax=pmax';
pvol=prod(Lmax-Lmin);

% Here is where the volumes are computed based on the MC assumption
% vol=pvol/nvol;

ns=nrsample*ones(nvol,1);

for ivol=1:nvol
    geom(ivol,1:Ldim)=psample(1:Ldim,ivol);
    geom(ivol,Ldim+1)=vol(ivol);
end

Qdim=qdim*nobs;
nq=sum(ns);
Qvals=zeros(nq,Qdim);

for ivol=1:nvol
    ip=sum(ns(1:ivol-1))+1;
    for ir=1:ns(ivol)
        iq=ip+ir-1;
        Qvals(iq,1:Qdim)=qsample(1:Qdim,ir,ivol);
    end
end

omean=qref(1:Qdim);
osigma=abs(cvar*omean);

nedge=nedge*ones(1,Qdim);


SIP_matfile=[output_directory,'/SIP_matfile.mat'];
fprintf('Saving SIP_matfile.mat \n')
save(SIP_matfile,...
     'Ldim','Lmin','Lmax','nvol','geom',...
     'Qdim','ns','Qvals',...
     'qmin','qmax','omean','osigma','nedge');

 
end

