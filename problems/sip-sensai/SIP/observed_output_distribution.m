function [pout,edges,nedge]=observed_output_distribution(edges,nedge,otype,ofname,omean,osigma,...
                                                         qdim,iprint,iplot)
%
%   ***  [pout]=observed_output_distribution(edges,nedge,otype,omean,osigma,...
%                                            qdim,iprint,iplot)   ***
%
%  Calculate the observed output distribution
%  Called by SIP_SENSAI
%
pout=zeros(qdim,max(nedge));

for idim=1:qdim
    
    % Uniform output distribution
    if otype(idim)==0 
        lower(idim)=omean(idim)-osigma(idim);
        upper(idim)=omean(idim)+osigma(idim);
        for ie=1:nedge(idim)-1
            x1=edges(idim,ie);
            x2=edges(idim,ie+1);
            if x1 >= lower(idim) & x2 <= upper(idim)
                 pout(idim,ie)=1;
            end
        end
        isum=sum(pout(idim,:));
        if isum ~= 0
            pout(idim,:)=pout(idim,:)/isum; 
        else
            fprintf('Divide by zero in approximate_pout \n')
            return
        end
    end
    
    % Normal output distribution
    if otype(idim)==1 
        for ie=1:nedge(idim)-1
            x1=edges(idim,ie);
            x2=edges(idim,ie+1);
            p1=0.5*(1+erf((x1-omean(idim))/sqrt(2*osigma(idim)^2)));
            p2=0.5*(1+erf((x2-omean(idim))/sqrt(2*osigma(idim)^2)));
            pout(idim,ie)=p2-p1;
        end
    end
        
    % Simulated output distribution
    if otype(idim)==3 
        % load fq      
        load(['../SENSAI_MuPad/',ofname,'.mat'])
        [nss,nr]=size(fq);
        % fq(omean(idim),1:nr)
        % edges(idim,1:nedge(idim))
        [n,bin]=histc( fq(omean(idim),1:nr), edges(idim,1:nedge(idim)) );
        nn=sum(n);
        for j=1:nedge(idim)
            pout(idim,j)=n(j)/nn;
        end
        % disp(pout)
    end
        
    if iprint >= 2
        fprintf('sum(pout)=%8.4f \n',sum(pout(idim,:)))
    end
    if iprint >= 6
        fprintf('pout for idim = %3i \n', idim)
        pout'
        fprintf('sum(pout) = %13.6e \n', sum(pout))
    end
    
    xplot=0.5*(edges(idim,1:nedge(idim)-1)+edges(idim,2:nedge(idim)));
    if iplot >= 2
        figure(5)
        subplot(1,qdim,idim), bar(xplot,pout(idim,1:nedge(idim)-1))
        title(['Observed pdf for QoI #',num2str(idim)])
        xlabel('Output')
        ylabel('Probability')
    end
    
end


end

