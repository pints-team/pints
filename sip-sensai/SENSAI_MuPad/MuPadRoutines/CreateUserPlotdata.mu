CreateUserPlotdata := proc(iplot_sensitivities,iplot_elasticities,
                           iplot_dxdp,iplot_dxdp_var,iplot_dxdp_param,
                           iplot_dxdz,
                           iplot_dqdp,iplot_dqdz,
                           iplot_cp, 
                           ilist,nilist,klist,nklist)

local fd, i;

begin

// Write plot data to user_plotdata.m
fd:=fopen("user_plotdata.m",Text, Write); 
fprint(Unquoted, fd, "function [iplot,ilist,klist]=user_plotdata"); 

fprint(Unquoted, fd, ""); 
fprint(Unquoted, fd, "iplot.sensitivities = ",iplot_sensitivities,";"); 
fprint(Unquoted, fd, "iplot.elasticities = ",iplot_elasticities,";"); 

fprint(Unquoted, fd, "iplot.dxdp.true = ",iplot_dxdp,";"); 
fprint(Unquoted, fd, "iplot.dxdp.var = ",iplot_dxdp_var,";"); 
fprint(Unquoted, fd, "iplot.dxdp.param = ",iplot_dxdp_param,";"); 

fprint(Unquoted, fd, "iplot.dxdz = ",iplot_dxdz,";");
 
fprint(Unquoted, fd, "iplot.dqdp = ",iplot_dqdp,";"); 
fprint(Unquoted, fd, "iplot.dqdz = ",iplot_dqdz,";"); 

fprint(Unquoted, fd, "iplot.cp = ",iplot_cp,";");
 
fprint(Unquoted, fd, ""); 
for i from 1 to nilist do;
    fprint(Unquoted, fd, "ilist(",i,") = ",ilist[i],";"); 
end_for; 

fprint(Unquoted, fd, ""); 
for i from 1 to nklist do;  
  fprint(Unquoted, fd, "klist(",i,") = ",klist[i],";"); 
end_for; 

fprint(Unquoted, fd, ""); 
fprint(NoNL, fd, "end"); 

fclose(fd);

end_proc;

