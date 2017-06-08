function plot_1D_projection(output_directory,nvol,geom,pinp,pmin,pmax,pedges,pnedge,qdim,idim)
%  ***   plot_1D_projection(output_directory,nvol,geom,pinp,pmin,pmax,pedges,pnedge,qdim,idim)    ***
%
%
%  Plot 1D projection through posteriori probability space
%

%   Histogram the samples according to the value of the parameters
%       n contains the number of samples in each OUTPUT bin
%       bin contains the OUTPUT bin number for each sample

iprint=2;

[n,bin]=histc(geom(:,idim),pedges(idim,1:pnedge(idim)));
if iprint >= 2
    fprintf('length(n) = %6i, sum(n) = %6i, length(bin) = %6i \n',...
        length(n),sum(n),length(bin))
end

% Compute input probabilities using individual and combined QoIs
ppsum(1:qdim+1,1:max(pnedge))=zeros(qdim+1,max(pnedge));

for jdim=1:qdim+1
    
    for ivol=1:nvol
        ppsum(jdim,bin(ivol))=ppsum(jdim,bin(ivol))+pinp(ivol,jdim);
    end
    
    % Plot computed input probabilities
%    plot_marginal_input_probabilities(output_directory,qdim,idim,jdim,pmin,pmax,pedges,pnedge,ppsum)

    ifig=100+idim;
    figure(ifig);
    subplot(1,qdim+1,jdim), plot(pedges(idim,1:pnedge(idim)),ppsum(jdim,1:pnedge(idim)),'o-','LineWidth',2)
    xlabel(['p(',num2str(idim),')'])
    ylabel('Probability')
    title(['Marginal probability for p(',num2str(idim),') using jdim=',num2str(jdim)])
    axis([0.99*pmin(idim),1.01*pmax(idim),0,0.35])

    figname=['-f',num2str(ifig)];
    filename=[output_directory,'/p',num2str(idim),'.eps'];
    print(figname,'-depsc',filename)

    if iprint >= 4
        print_marginal_input_probabilities('screen',idim,jdim,pnedge,ppsum);
    end
    
    % Print computed input probabilities
    if jdim==qdim+1
        print_marginal_input_probabilities('screen',idim,jdim,pnedge,ppsum);
        print_marginal_input_probabilities(output_directory,idim,jdim,pnedge,ppsum);
    end
    
end
    
end

