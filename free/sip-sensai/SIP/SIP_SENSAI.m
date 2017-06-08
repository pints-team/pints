function [pinp]=SIP_SENSAI(input_source,output_source,ExampleName,JobName,iprint,iplot)
%   ***  [pinp]=SIP_SENSAI(input_source,output_source,ExampleName,JobName,iprint,iplot)   ***
%
% Inputs
%  input_source  = directory containing inputfile created by SENSAI
%  output_source = directory to store the output
%  ExampleName = subdirectory name for input and output
%  JobName = subsubdirectory name for input and output
%  iprint = print level
%  iplot  = plot level
%
% Variables for define_inputs
% ---------------------------
%  Ldim       = dimension of the input (parameter) space
%  Lmin(idim) = lower bounds of input (parameter) space, idim = 1,...,Ldim
%  Lmax(idim) = upper bounds of input (parameter) space, idim = 1,...,Ldim
%  Lres(idim) = number of intervals of the input (parameter) space, idim = 1,...,Ldim
%  qdim       = dimension of the output space
%
%
% Variables for calculate_samples
% -------------------------------
%  nvol = number of volumes used to discretize the input (parameter) space
%  geom(ivol,1) = x-coordinates of ith input (parameter) volume, ivol=1,...,nvol
%  geom(ivol,2) = y-coordinates of ith input (parameter) volume, ivol=1,...,nvol 
%  geom(ivol,3) = z-coordinates of ith input (parameter) volume, ivol=1,...,nvol
%  geom(ivol,Ldim+1) = volume of ith input (parameter) volume, ivol=1,...,nvol
%  Qvals(ivol,idim) = value of realization ivol = 1,...,nvol, idim=1,...,qdim
%
% Variables for define_output
% ---------------------------
%  fmin(jdim)= lower bounds of output space, jdim = 1,...,qdim 
%  fmax(jdim)= upper bounds of output space, jdim = 1,...,qdim 
%  mean(jdim)= mean of output distribution, jdim = 1,...,qdim  
%  sigma(jdim)= variance of output distribution, jdim = 1,...,qdim  
%  fnedge(jdim)= number of intervals of the output space, jdim = 1,...,qdim
%  fedges(jdim,1:fnedge(jdim))= edges of the histogram for the output space
%  pnedge(idim)= number of intervals of the input space, idim=1,...,pdim


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define discretization of input space and forward simulations                 %   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rehash path
format long
% Load simulation data

fprintf('Read input/output map \n')
[input_directory,geom,Qvals,Ldim,nvol,ns,nq]=Input_Output_map(input_source,ExampleName,JobName);

fprintf('Input parameters from SIP_SENSAI_input \n')
[Qindex,fmin,fmax,fnedge,otype,ofname,omean,osigma,pnedge]=SIP_SENSAI_input;

fprintf('Check inputs and create qvals \n')
[qvals,qdim]=Input_checks(Qindex,Qvals,nq,fmin,fmax,fnedge,otype,ofname,omean,osigma,pnedge);

fprintf('Make output directory \n')
[output_directory]=Make_Output_Directory(output_source,ExampleName,JobName,otype,ofname,Qindex,qdim);


% Construct pointer array from function evaluation to volume
fprintf('Construct pointer array \n')
vpoint=zeros(nq,1);
ip2=0;
for ivol=1:nvol
    ip1=ip2+1;
    ip2=ip1+ns(ivol)-1;
    vpoint(ip1:ip2,1)=ivol;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Analyze forward simulations                                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cond_prob_store=zeros(nvol,max(fnedge),qdim);

for idim=1:qdim
    
    fprintf('Compute conditional probabilities for QoI %5i \n', idim)
    
    % For plotting purposes only
    nex=ceil(sqrt(fnedge(idim)));
    ney=ceil(fnedge(idim)/nex);
    
    % Establish the "fedges" for the histogram
    fedges(idim,1:fnedge(idim))=linspace(fmin(idim),fmax(idim),fnedge(idim));
    
    %   Histogram the output
    %       n contains the number of samples in each OUTPUT bin
    %       bin contains the OUTPUT bin number for each sample
    
    [n,bin]=histc(qvals(:,idim),fedges(idim,1:fnedge(idim)));
    fprintf('\nlength(n) = %6i, sum(n) = %6i, length(bin) = %6i \n',...
        length(n),sum(n),length(bin))
    
    plot_samples_histogram(output_directory,fedges(idim,1:fnedge(idim)),n,bin,...
                           Qindex,qdim,idim,iprint,iplot)
    
    % Accumulate the number of times each input volume contributes
    % to each output bin
    Lmap=sparse(nvol,fnedge(idim)-1);
    Mmap=sparse(nvol,fnedge(idim)-1);
    Mmap_sum=zeros(1,fnedge(idim)-1);
    Mvmap=sparse(nvol,fnedge(idim)-1);
    Mvmap_sum=zeros(1,fnedge(idim)-1);
    
    cond_prob=sparse(nvol,fnedge(idim)-1);
    
    for iq=1:nq
        if bin(iq)>0
            Lmap(vpoint(iq),bin(iq))=Lmap(vpoint(iq),bin(iq))+1;
        end
    end
    
    if abs(nq-sum(sum(Lmap,1)))>0
        fprintf('Houston: We have a problem in columns of Lmap \n')
        return
    end
    if max(abs(sum(Lmap,2)-ns))>0
        fprintf('Houston: We have a problem in rows of Lmap \n')
        return
    end
    
    if iprint >= 6
        for ivol=1:nvol
            fprintf('ivol = %5i \n', ivol)
            disp(find(Lmap(ivol,:)))
        end
        pause
    end
    
    % Analyze the input to output map column by column, i.e., by output
    for ie=1:fnedge(idim)-1
        Mmap(1:nvol,ie)=Lmap(1:nvol,ie);
        Mmap_sum(ie)=sum(Mmap(1:nvol,ie));
        if Mmap_sum(ie) > 1e-16
            Mmap(1:nvol,ie)=Mmap(1:nvol,ie)/Mmap_sum(ie);
        end
        
        Mvmap(1:nvol,ie)=Mmap(1:nvol,ie).*geom(1:nvol,Ldim+1);
        Mvmap_sum(ie)=sum(Mvmap(1:nvol,ie));
        
        if Mmap_sum(ie) == 0
            cond_prob(1:nvol,ie)=zeros(nvol,1);
        else
            cond_prob(1:nvol,ie)=Mvmap(1:nvol,ie)/Mvmap_sum(ie);
        end
        
        if iplot >= 4
            plot_contour(geom(1:nvol,1:Ldim),Ldim,idim,fedges(idim,1:fnedge(idim)),ie,...
                nex,ney,cond_prob(:,ie))
        end
        
    end
    
    % Store conditional probabilities
    fprintf('Computed conditional probabilities for idim = %4i \n', idim)
    cond_prob_store(1:nvol,1:fnedge(idim)-1,idim)=cond_prob;
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Given observed output distribution, distribute probability over input space  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Compute probability of observed distribution on output bins
[pout,fedges,fnedge]=observed_output_distribution(fedges,fnedge,otype,ofname,omean,osigma,...
                                                  qdim,iprint,iplot);
                                              
if iprint > 4
    fprintf('pout \n')
    disp(pout)
end

% Distribute probabilities over generalized contours
fprintf('\nDistribute probabilities over generalized contours \n')
pinp=zeros(nvol,qdim+1);

for idim=1:qdim
    cp=cond_prob_store(1:nvol,1:fnedge(idim)-1,idim);
    % fprintf('cp is a sparse matrix (0/1): %3i \n', issparse(cp))
    
    for ivol=1:nvol
        pinp(ivol,idim)=pinp(ivol,idim)+cp(ivol,1:fnedge(idim)-1)*pout(idim,1:fnedge(idim)-1)';
    end
    sump=sum(pinp(1:nvol,idim));
    fprintf('sum(pinp(1:nvol,%3i) = %13.6e \n', idim, sump)
    
    % pinp(1:nvol,idim) = pinp(1:nvol,idim)/sum(pinp(1:nvol,idim));
    if abs(sump-1.0)>1e-06
        fprintf('Houston: We have a problem sump for idim=%4i \n', idim)
        %return
    end
end

% Assume QoIs are independent and create input probabilities based on all QoIs
fprintf('\nCombine independent QoIs\n')
pinp(1:nvol,qdim+1)=ones(nvol,1);
for idim=1:qdim
    pinp(1:nvol,qdim+1)=pinp(1:nvol,qdim+1).*pinp(1:nvol,idim);
end
%
sump=sum(pinp(1:nvol,qdim+1));
pinp(1:nvol,qdim+1)=pinp(1:nvol,qdim+1)/sump;
%
sump=sum(pinp(1:nvol,qdim+1));
fprintf('idim=%4i, sum(pinp(1:nvol,qdim+1)) = %13.6e \n', qdim+1, sump)
if abs(sump-1.0)>1e-06
    fprintf('Houston: We have a problem sump for idim=%3i \n', qdim+1)
    return
end

% Construct scatter plots of computed probabilities of input parameters
for jdim=1:qdim+1
    plot_input_probabilities(output_directory,pinp(1:nvol,jdim),Ldim,jdim,geom(1:nvol,1:Ldim),nvol)
end

% Establish the "pedges" for the histogram
for idim=1:Ldim   
    pmin(idim)=min(geom(:,idim));
    pmax(idim)=max(geom(:,idim));   
    fprintf('\nCalculate input probability distributions for p(%4i) \n', idim)
    pedges(idim,1:pnedge(idim))=linspace(0.99*pmin(idim),1.01*pmax(idim),pnedge(idim));  
end

% Compute and plot projections on two parameter axes
for idim=1:Ldim
    plot_1D_projection(output_directory,nvol,geom,pinp,pmin,pmax,pedges,pnedge,qdim,idim)   
end

% Compute and plot projections on to parameter planes
if Ldim==3
    plot_2D_projection(output_directory,nvol,geom,pinp,pmin,pmax,pedges,pnedge,qdim,1,2)
    plot_2D_projection(output_directory,nvol,geom,pinp,pmin,pmax,pedges,pnedge,qdim,1,3)
    plot_2D_projection(output_directory,nvol,geom,pinp,pmin,pmax,pedges,pnedge,qdim,2,3)
end


%[s,mess,messid]=mkdir(output_directory);
if ~(strcmp(input_directory, output_directory))
    fprintf(['Copy SIP_matfile from', input_directory, ' to ', output_directory '\n'])
    copyfile([input_directory,'/SIP_matfile.mat'], [output_directory,'/SIP_matfile.mat'])
end
copyfile('SIP_SENSAI_input.m', [output_directory,'/SIP_SENSAI_input.m'])
filename=[output_directory,'/SIP_SENSAI_output.mat']
save(filename,'cond_prob_store', 'pinp')

end
