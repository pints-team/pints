function plot_warnings(iplot,solution_only)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                                                  %
%     Combinational Warning Optiions                                                               %
%                                                                                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plot_select = 0;
organize_select = 0;
xwrtp_select = 0;

if(iplot.sensitivities == 1 || iplot.elasticities == 1)
    plot_select = 1;
end

if(iplot.dxdp.var == 1 || iplot.dxdz == 1 || iplot.dqdp == 1 || iplot.dqdz == 1 || iplot.cp == 1 || iplot.dxdp.param == 1 || iplot.dxdp.true)
    organize_select = 1;
end

if(iplot.dxdp.var == 1 || iplot.dxdp.param == 1)
    xwrtp_select = 1;
end

% Display a warning if solutions only is selected and how to organize sensitivity plots is also selected
if(solution_only == 1 && (plot_select == 1 || organize_select == 1 || xwrtp_select == 1))
    fprintf('\n\nWarning: You have selected to compute the solutions only, but have also specified sensitivity and/or elasticity plots.  Default is to compute solutions only.\n');
    msgbox('Warning: You have selected to compute the solutions only, but have also specified sensitivity and/or elasticity plots.  Default is to compute solutions only.');
end

if(solution_only == 0)
    %Display a warning if type of plot is specified, but how to organize is not
    if(plot_select == 1 && organize_select == 0)
        fprintf('\n\nWarning: You have specified a sensitivity or elasticity but have not specified how to organize them. Default is to plot solutions only.\n');
        msgbox('Warning: You have specified a sensitivity or elasticity but have not specified how to organize them. Default is to plot solutions only.');
    end

    %Display a warning if how to organize is specified, but type of plot is not
    if(plot_select == 0 && organize_select == 1)
        fprintf('\n\nWarning: You have not specified a sensitivity or elasticity but have specified how to organize them.  Default is to plot solutions only.\n');
        msgbox('Warning: You have not specified a sensitivity or elasticity but have specified how to organize them.  Default is to plot solutions only.');
    end

    %Display a warning if xwrtp is selected by not how to organize it
    if(plot_select == 1 && iplot.dxdp.true == 1 && xwrtp_select == 0)
        fprintf('\n\nWarning: You have specified plotting x with respect to p, but not how to organize.  Default is to organize by state.\n');
        msgbox('Warning: You have specified plotting x with respect to p, but not how to organize.  Default is to organize by state.');
    end

    %Display a warning if xwrtp is not selected but how to organize it is
    if(plot_select == 1 && iplot.dxdp.true == 0 && xwrtp_select == 1)
        fprintf('\n\nWarning: You have specified how to organize x with respect to p plots, but have not checked x with respect to p.  Default is to plot as specified.\n');
        msgbox('Warning: You have specified how to organize x with respect to p plots, but have not checked x with respect to p.  Default is to plot as specified.');
    end
        
end
