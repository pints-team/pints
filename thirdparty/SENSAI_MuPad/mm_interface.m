function mm_interface(gparam,gvec,qoi,cp,qtype,stype,NextGen,x0,p,imap,R0_only)
%
% Runs maple procedures from matlab
tic;
fprintf('\nmm_interface \n')
qoi_only=0;
if qoi_only==1
    fprintf('\n\nWARNING: SENSAI will produce QoI.m and cp.m ONLY \n\n')
end

xdim=gparam(1);
kdim=gparam(2);
qdim=gparam(3);

fprintf('mm_interface: xdim =  %3i \n', xdim)
fprintf('mm_interface: kdim =  %3i \n', kdim)
for i=1:xdim
    %fprintf('mm_interface: g(%3i) is a string (true/false=1/0) %3i \n', i,ischar(g(i,:)))
    fprintf('mm_interface: g(%3i) =  %s \n', i,gvec(i,1:length(gvec(i,:))) )
end
fprintf('mm_interface: cp =  %s \n', cp)

gtotal=gvec(1,1:length(gvec(1,:)));
for i=2:xdim
    gtotal=strcat(gtotal,';',gvec(i,1:length(gvec(i,:))));
end
gtotal=strcat(' " ',gtotal,' " ')
%fprintf('mm_interface: gtotal is a string (true/false=1/0) %3i \n', ischar(gtotal))

% New gvec MuPad file
fprintf('mm_interface: call gvec.mu \n');
read(symengine, 'MuPadRoutines/gvec.mu');
feval(symengine, 'gvec',gparam,gtotal);

if NextGen(1) ~= 0    

    if R0_only == 1
        % Create Next Generation Matrix with MuPad
        fprintf('mm_interface: call r0_only.mu \n');
        read(symengine, 'MuPadRoutines/r0_only.mu');
        feval(symengine, 'r0_only',NextGen,gparam,gtotal,x0,p,imap);
    else
        % Create Next Generation Matrix with MuPad
        fprintf('mm_interface: call r0.mu \n');
        read(symengine, 'MuPadRoutines/r0.mu');
        feval(symengine, 'r0',NextGen,gparam,gtotal,x0,p,imap);
    end
    R0_time = toc;
    if R0_time > 60 
        fprintf(['R0 took ',num2str(R0_time/60),' MINUTES to compute.\n']);
    else
        fprintf(['R0 took ',num2str(R0_time),' SECONDS to compute.\n']);
    end

end
    
if(qtype==1 & stype >= 2)
    
    if(qoi_only==0)
        
        % dgvec_dxvec MuPad files
        fprintf('mm_interface: call dgvec_dxvec.mu \n');
        read(symengine, 'MuPadRoutines/dgvec_dxvec.mu');
        feval(symengine, 'dgvec_dxvec',gparam,gtotal);
        
        % dgvec_dparam MuPad files
        fprintf('mm_interface: call dgvec_dparam.mu \n');
        read(symengine, 'MuPadRoutines/dgvec_dparam.mu');
        feval(symengine, 'dgvec_dparam',gparam,gtotal);
        
        if imap > 1
            % d2gvec_dxvec2 MuPad files
            fprintf('mm_interface: call d2gvec_dxvec2.mu \n');
            read(symengine, 'MuPadRoutines/d2gvec_dxvec2.mu');
            feval(symengine, 'd2gvec_dxvec2',gparam,gtotal);
            
            % d2gvec_dxvec_dparam MuPad files
            fprintf('mm_interface: call d2gvec_dxvec_dparam.mu \n');
            read(symengine, 'MuPadRoutines/d2gvec_dxvec_dparam.mu');
            feval(symengine, 'd2gvec_dxvec_dparam',gparam,gtotal);
        end
        
    end
    
    % QoI MuPad files
    for i=1:qdim
        %fprintf('mm_interface: g(%3i) is a string (true/false=1/0) %3i \n', i,ischar(g(i,:)))
        fprintf('mm_interface: q(%3i) =  %s \n', i,qoi(i,1:length(qoi(i,:))) )
    end

    qtotal=qoi(1,1:length(qoi(1,:)));
    for i=2:qdim
        qtotal=strcat(qtotal,';',qoi(i,1:length(qoi(i,:))));
    end
    qtotal=strcat(' " ',qtotal,' " ')
    %fprintf('mm_interface: qtotal is a string (true/false=1/0) %3i \n', ischar(qtotal))    
   
    fprintf('mm_interface: call qoi.mu \n');
    read(symengine, 'MuPadRoutines/qoi.mu');
    feval(symengine, 'qoi',gparam,qtotal);
    
    % cp MuPad files
    fprintf('mm_interface: call cp.mu \n');
    read(symengine, 'MuPadRoutines/cp.mu');
    feval(symengine, 'cp',gparam,cp);
    
end

toc;
end
