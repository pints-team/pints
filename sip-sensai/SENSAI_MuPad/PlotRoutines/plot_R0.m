function plot_R0(output_directory,R0,dR0dp,elR0p,dR0dc,elR0c,klist,iplot,R0_only)
%
%   ***   plot_R0(output_directory,t,q,dqdparam,xdim,kdim)   ***
%

kdim = length(klist);

ktick = [''];
for k = 1:kdim
    if klist(k) < 10
        ktick(k,:) = ['p0',num2str(klist(k))];
    else
        ktick(k,:) = ['p',num2str(klist(k))];
    end
end


if  R0_only == 0
    if(iplot.cp == 1)
        if(iplot.sensitivities == 1)
            % Create a bar plot with R0 and sensitivities of p, cp
            figure(800)
            subplot(2,2,1), bar(R0), set(gca,'XTickLabel',{''}), title('R0')
            subplot(2,2,2), bar(dR0dc), set(gca,'XTickLabel',{''}), title('Sensitivity of R0 wrt cp'), %xlabel('dR0dz');
            subplot(2,2,3:4), bar(dR0dp(klist)), set(gca,'XTick',1:kdim), set(gca,'XTickLabel',ktick), title('Sensitivity of R0 wrt p'), %xlabel('dR0dp');
        end
        if(iplot.elasticities == 1)
            % Create a bar plot with R0 and elasticities of p, cp
            figure(850)
            subplot(2,2,1), bar(R0), set(gca,'XTickLabel',{''}), title('R0')
            subplot(2,2,2), bar(elR0c), set(gca,'XTickLabel',{''}), title('Elasticity of R0 wrt cp'), %xlabel('elR0z');
            subplot(2,2,3:4), bar(elR0p(klist)), set(gca,'XTick',1:kdim), set(gca,'XTickLabel',ktick), title('Elasticity of R0 wrt p'), %xlabel('elR0p');
        end
        if(iplot.sensitivities == 0 && iplot.elasticities == 0)
            % Create a bar plot with just R0 and cp
            figure(800)
            subplot(1,2,1), bar(R0), set(gca,'XTickLabel',{''}), title('R0')
            subplot(1,2,2), bar(elR0c), set(gca,'XTickLabel',{''}), title('Elasticity of R0 wrt cp'), %xlabel('elR0z');
        end
    else
        if(iplot.sensitivities == 1)
            % Create a bar plot with R0 and sensitivities of p
            figure(800)
            subplot(2,2,1), bar(R0), set(gca,'XTickLabel',{''}), title('R0')
            subplot(2,2,3:4), bar(dR0dp(klist)), set(gca,'XTick',1:kdim), set(gca,'XTickLabel',ktick), title('Sensitivity of R0 wrt p'), %xlabel('dR0dp');
        end
        if(iplot.elasticities == 1)
            % Create a bar plot with R0 and elasticities of p
            figure(850)
            subplot(2,2,1), bar(R0), set(gca,'XTickLabel',{''}), title('R0')
            subplot(2,2,3:4), bar(elR0p(klist)), set(gca,'XTick',1:kdim), set(gca,'XTickLabel',ktick), title('Elasticity of R0 wrt p'), %xlabel('elR0p');
        end
        if(iplot.sensitivities == 0 && iplot.elasticities == 0)
            % Create a bar plot with just R0
            figure(800)
            bar(R0), set(gca,'XTickLabel',{''}), title('R0')
        end
        
    end
    
else
    % Create a bar plot with just R0
    figure(800)
    bar(R0), set(gca,'XTickLabel',{''}), title('R0')
end

if iplot.sensitivities == 1
    figname='-f800';
    filename=[output_directory,'\R0_senstivities.eps'];
    print(figname,'-depsc',filename)
end

if(iplot.elasticities == 1 && R0_only == 0)
    figname='-f850';
    filename=[output_directory,'\R0_elasticities.eps'];
    print(figname,'-depsc',filename)
end


end

