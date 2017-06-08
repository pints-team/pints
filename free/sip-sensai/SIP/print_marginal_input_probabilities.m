function print_marginal_input_probabilities(output_directory,idim,jdim,pnedge,ppsum)
%
%  ***   print_marginal_input_probabilities(output_directory,idim,jdim,pnedge,ppsum)  ***
%
%  Print the computed marginal input probabilities
%  Called by SIP_SENSAI
%

if strcmp(output_directory,'screen')
    
    fprintf('Marginal probability for p(%4i) \n', idim)
    if jdim==2
        fprintf('  SUM  %13.6e  %13.6e \n', ...
            sum(ppsum(1,:)),sum(ppsum(2,:)) );
    elseif jdim==3
        fprintf('  SUM  %13.6e  %13.6e   %13.6e \n',...
            sum(ppsum(1,:)),sum(ppsum(2,:)),sum(ppsum(3,:)) );
    elseif jdim==4
        fprintf('  SUM  %13.6e  %13.6e  %13.6e  %13.6e \n',...
            sum(ppsum(1,:)),sum(ppsum(2,:)),sum(ppsum(3,:)),sum(ppsum(4,:)) );
    end
    
    if jdim==3
        fprintf('  jdim    p(QoI1)        p(QoI2)         p(QoI1-2)      ratio \n')
        for ii=1:pnedge(idim)
            tt1=ppsum(1,ii)*ppsum(2,ii);
            tt2=0;
            if ppsum(3,ii) > 1e-16
                tt2=(ppsum(1,ii)*ppsum(2,ii))/ppsum(3,ii);
            end
            fprintf('%5i  %13.6e  %13.6e   %13.6e  %13.6e \n', ...
                ii, ppsum(1,ii), ppsum(2,ii), tt1, tt2)
        end
    end
      
else
    
    if jdim==3
        filename=[output_directory,'/marginal_input_probabilities'];
        fid=fopen(filename,'a');
        fprintf(fid, 'Marginal probability for p(%4i) \n', idim);
        fprintf(fid, '  jdim    p(QoI1)        p(QoI2)         p(QoI1-2)      ratio \n');
        for ii=1:pnedge(idim)
            tt1=ppsum(1,ii)*ppsum(2,ii);
            tt2=0;
            if ppsum(3,ii) > 1e-16
                tt2=(ppsum(1,ii)*ppsum(2,ii))/ppsum(3,ii);
            end
            fprintf(fid, '%5i  %13.6e  %13.6e   %13.6e  %13.6e \n', ...
                ii, ppsum(1,ii), ppsum(2,ii), tt1, tt2);
        end
        fclose(fid);
    end
    
    
end


end

