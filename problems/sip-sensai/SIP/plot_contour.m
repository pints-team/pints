function plot_contour(p,Ldim,idim,edges,ie,nex,ney,cond_prob)
%
%   ***  plot_contour(p,Ldim,idim,edges,ie,nex,ney,prob)  ***
%
%

fprintf('plot_contour, Ldim= %4i, idim = %4i, ie = %4i, \n', Ldim,idim,ie)

figure(29+idim)
if Ldim == 3
    subplot(nex,ney,ie), scatter3(p(:,1),p(:,2),p(:,3),25,cond_prob,'*')
    title(['Output ',num2str(ie),': ', num2str(edges(ie)),' - ',num2str(edges(ie+1))])
    xlabel('x')
    ylabel('y')
    zlabel('z')
elseif Ldim ==2
    subplot(nex,ney,ie), scatter(p(:,1),p(:,2),25,cond_prob,'*')
    title(['Output ',num2str(ie),': ', num2str(edges(ie)),' - ',num2str(edges(ie+1))])
    xlabel('x')
    ylabel('z')
end

end

