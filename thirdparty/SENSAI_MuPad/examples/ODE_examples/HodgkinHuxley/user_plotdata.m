function [iplot,ilist,klist]=user_plotdata

iplot.sensitivities = 1;
iplot.elasticities = 0;
iplot.dxdp.true = 0;
iplot.dxdp.var = 1;
iplot.dxdp.param = 0;
iplot.dxdz = 0;
iplot.dqdp = 1;
iplot.dqdz = 1;
iplot.cp = 0;

ilist(1) = 1;
ilist(2) = 2;
ilist(3) = 3;
ilist(4) = 4;

klist(1) = 1;
klist(2) = 2;
klist(3) = 3;

end