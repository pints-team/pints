function [iplot,ilist,klist]=user_plotdata

iplot.sensitivities = 1;
iplot.elasticities = 1;
iplot.dxdp.true = 1;
iplot.dxdp.var = 0;
iplot.dxdp.param = 1;
iplot.dxdz = 1;
iplot.dqdp = 1;
iplot.dqdz = 0;
iplot.cp = 0;

ilist(1) = 1;

klist(1) = 1;
klist(2) = 2;

end