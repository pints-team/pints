function plot_Voronoi(pmin,pmax,psim,psample,nsample,count,vol2sim)
%  ***   plot_Voronoi(pmin,pmax,psim,psample,nsample,count,vol2sim)   ***
%
% Plot Voronoi cells
%

figure(55)
hold off; clf
[kdim,nsim]=size(psim);
   
if kdim==2
    axis([0.99*pmin(1) 1.01*pmax(1) 0.99*pmin(2) 1.01*pmax(2)])
    axis square
    title('Voronoi cells')
    xlabel('Parameter 1')
    ylabel('Parameter 2')
    hold on
    for ivol=1:nsample
        for j=1:count(ivol)
            if mod(ivol,4)==1
                plot(psample(1,ivol),psample(2,ivol),'sm')
                plot(psim(1,vol2sim(ivol,j)),psim(2,vol2sim(ivol,j)),'om')
            elseif mod(ivol,4)==2
                plot(psample(1,ivol),psample(2,ivol),'sg')
                plot(psim(1,vol2sim(ivol,j)),psim(2,vol2sim(ivol,j)),'og')
            elseif mod(ivol,4)==3
                plot(psample(1,ivol),psample(2,ivol),'sc')
                plot(psim(1,vol2sim(ivol,j)),psim(2,vol2sim(ivol,j)),'oc')
            else
                plot(psample(1,ivol),psample(2,ivol),'sk')
                plot(psim(1,vol2sim(ivol,j)),psim(2,vol2sim(ivol,j)),'ok')
            end
        end
    end
end

if kdim==3
    axis([0.99*pmin(1) 1.01*pmax(1) 0.99*pmin(2) 1.01*pmax(2) 0.99*pmin(3) 1.01*pmax(3)])
    axis square
    title('Voronoi cells')
    xlabel('Parameter 1')
    ylabel('Parameter 2')
    zlabel('Parameter 3')
    hold on
    for ivol=1:nsample
        for j=1:count(ivol)
            if mod(ivol,4)==1
                scatter3(psample(1,ivol),psample(2,ivol),psample(3,ivol),25,[1 0 0],'s','LineWidth',3)
                scatter3(psim(1,vol2sim(ivol,j)),psim(2,vol2sim(ivol,j)),psim(3,vol2sim(ivol,j)),25,[1 0 0],'o')
            elseif mod(ivol,4)==2
                scatter3(psample(1,ivol),psample(2,ivol),psample(3,ivol),25,[0 1 0],'s','LineWidth',3)
                scatter3(psim(1,vol2sim(ivol,j)),psim(2,vol2sim(ivol,j)),psim(3,vol2sim(ivol,j)),25,[0 1 0],'o')
            elseif mod(ivol,4)==3
                scatter3(psample(1,ivol),psample(2,ivol),psample(3,ivol),25,[0 0 1],'s','LineWidth',3)
                scatter3(psim(1,vol2sim(ivol,j)),psim(2,vol2sim(ivol,j)),psim(3,vol2sim(ivol,j)),25,[0 0 1],'o')
            else
                scatter3(psample(1,ivol),psample(2,ivol),psample(3,ivol),25,[1 1 1]/3,'s','LineWidth',3)
                scatter3(psim(1,vol2sim(ivol,j)),psim(2,vol2sim(ivol,j)),psim(3,vol2sim(ivol,j)),25,[1 1 1]/3,'o')
            end
        end
    end
end

end
