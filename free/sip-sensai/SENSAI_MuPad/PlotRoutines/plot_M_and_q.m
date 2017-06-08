function plot_M_and_q(output_directory,nt,t,q,Mstore,Sstore,Mdim,Sdim,Fdim);
%

global NCOLUMNS

nklist=Fdim;
ncsubk=NCOLUMNS(nklist);
nrsubk=ceil(nklist/ncsubk);


for i=1:Fdim
    M(i,:)=Mstore(1,i,:);
    S(i,:)=Sstore(1,i,:);
end

% Plot components of the eigenvector, V_1
ifig=15500;
figure(ifig);
hold on
for i=1:Fdim
    plot( t(2:nt),M(i,2:nt),'-', 'LineWidth',2)
end
for j=1:min(Mdim,Fdim)
   plot( t(2:nt),S(1,2:nt),':', 'LineWidth',2)
end
plot(t,q,'--m','LineWidth',2)
title('M components')
figname=['-f',num2str(ifig)];
% filename=[output_directory,'/M_components_and_q.eps'];
% print(figname,'-depsc',filename)

end

