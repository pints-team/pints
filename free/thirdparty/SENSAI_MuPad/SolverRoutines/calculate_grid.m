function [pgrid,vgrid] = calculate_grid(ns,rs,kdim,pmin,pmax)
%  ***   [pgrid,vgrid] = calculate_grid(,ns,rs,kdim,pmin,pmax)   ***
%
% Constructs the grid in parameter space at which the map is computed
%

ms=floor(ns/2);
nsample=prod(ns);
fprintf('Number of samples = %8i \n', nsample)
pgrid=zeros(kdim,nsample);
vgrid=zeros(nsample,1);


regular_grid=1; 

if regular_grid==0
    rr=rand(kdim,nsample);
    for idim=1:kdim;
        pgrid(idim,1:nsample)=pmin(idim)+rr(idim,1:nsample)*(pmax(idim)-pmin(idim));
    end
end

if regular_grid==1
    if kdim==2     
              
        p1=stretch_grid(pmin(1),pmax(1),ms(1),rs);
        p2=stretch_grid(pmin(2),pmax(2),ms(2),rs);
        
        for i=1:ns(1)
            for j=1:ns(2)
                k=(i-1)*ns(2)+j;
                pgrid(1,k)=p1(i);
                pgrid(2,k)=p2(j);
            end
        end
        
        for i=1:ns(1)
            if i==1
                pleft(1)=pmin(1);
            else
                pleft(1)=p1(i-1);
            end
            if i==ns(1)
                pright(1)=pmax(1);
            else
                pright(1)=p1(i+1);
            end
            dp(1)=(pright(1)-pleft(1));
            
            for j=1:ns(2)
                if j==1
                    pleft(2)=pmin(2);
                else
                    pleft(2)=p2(j-1);
                end
                if j==ns(2)
                    pright(2)=pmax(2);
                else
                    pright(2)=p2(j+1);
                end
                k=(i-1)*ns(2)+j;
                
                dp(1)=(pright(1)-pleft(1));
                dp(2)=(pright(2)-pleft(2));
                vgrid(k)=prod(dp);
                
            end
        end
        
    elseif kdim==3
        
        p1=stretch_grid(pmin(1),pmax(1),ms(1),rs);
        p2=stretch_grid(pmin(2),pmax(2),ms(2),rs);
        p3=stretch_grid(pmin(3),pmax(3),ms(3),rs);
        
        for i=1:ns(1)
            for j=1:ns(2)
                for k=1:ns(3)
                    l=(i-1)*ns(2)*ns(3)+(j-1)*ns(3)+k;
                    pgrid(1,l)=p1(i);
                    pgrid(2,l)=p2(j);
                    pgrid(3,l)=p3(k);
                end
            end
        end
        
        for i=1:ns(1)
            if i==1
                pleft(1)=pmin(1);
            else
                pleft(1)=(p1(i-1)+p1(i))/2;
            end
            if i==ns(1)
                pright(1)=pmax(1);
            else
                pright(1)=(p1(i)+p1(i+1))/2;
            end
            dp(1)=pright(1)-pleft(1);
            
            for j=1:ns(2)
                if j==1
                    pleft(2)=pmin(2);
                else
                    pleft(2)=(p2(j-1)+p2(j))/2;
                end
                if j==ns(2)
                    pright(2)=pmax(2);
                else
                    pright(2)=(p2(j)+p2(j+1))/2;
                end
                dp(2)=pright(2)-pleft(2);
                
                for k=1:ns(3)
                    if k==1
                        pleft(3)=pmin(3);
                    else
                        pleft(3)=(p3(k-1)+p3(k))/2;
                    end
                    if k==ns(3)
                        pright(3)=pmax(3);
                    else
                        pright(3)=(p3(k)+p3(k+1))/2;
                    end
                    dp(3)=pright(3)-pleft(3);
                    
                    l=(i-1)*ns(2)*ns(3) + (j-1)*ns(3)+k;
                    vgrid(l)=prod(dp);
                    
                end
            end
        end
    end
    
end
    
    
    % elseif kdim==14
    %
    %     for i=1:kdim-1
    %         s(i)=prod(ns(i+1:kdim));
    %     end
    %
    %     for i=1:kdim
    %         if ns(i)>1
    %             pin(i)=(pmax(i)-pmin(i))/(2*ns(i));
    %             p(i,1:ns(i))=linspace(pmin(i)+pin(i),pmax(i)-pin(i),ns(i));
    %         else
    %             p(i,1)=(pmin(i)+pmax(i))/2;
    %         end
    %     end
    %
    %     for i1=1:ns(1)
    %         for i2=1:ns(2)
    %             for i3=1:ns(3)
    %                 for i4=1:ns(4)
    %                     for i5=1:ns(5)
    %
    %                         for i6=1:ns(6)
    %                             for i7=1:ns(7)
    %                                 for i8=1:ns(8)
    %                                     for i9=1:ns(9)
    %                                         for i10=1:ns(10)
    %
    %                                             for i11=1:ns(11)
    %                                                 for i12=1:ns(12)
    %                                                     for i13=1:ns(13)
    %                                                         for i14=1:ns(14)
    %
    %                                                             l=(i1-1)*s(1)+(i2-1)*s(2)+(i3-1)*s(3)+(i4-1)*s(4)+(i5-1)*s(5)...
    %                                                                 +(i6-1)*s(6)+(i7-1)*s(7)+(i8-1)*s(8)+(i9-1)*s(9)+(i10-1)*s(10)...
    %                                                                 +(i11-1)*s(11)+(i12-1)*s(12)+(i13-1)*s(13)+i14;
    %
    %                                                             pgrid(1,l)=p(1,i1);
    %                                                             pgrid(2,l)=p(2,i2);
    %                                                             pgrid(3,l)=p(3,i3);
    %                                                             pgrid(4,l)=p(4,i4);
    %                                                             pgrid(5,l)=p(5,i5);
    %
    %                                                             pgrid(6,l)=p(6,i6);
    %                                                             pgrid(7,l)=p(7,i7);
    %                                                             pgrid(8,l)=p(8,i8);
    %                                                             pgrid(9,l)=p(9,i9);
    %                                                             pgrid(10,l)=p(10,i10);
    %
    %                                                             pgrid(11,l)=p(11,i11);
    %                                                             pgrid(12,l)=p(12,i12);
    %                                                             pgrid(13,l)=p(13,i13);
    %                                                             pgrid(14,l)=p(14,i14);
    %
    %                                                         end
    %                                                     end
    %                                                 end
    %                                             end
    %                                         end
    %
    %                                     end
    %                                 end
    %                             end
    %                         end
    %                     end
    %
    %                 end
    %             end
    %         end
    %     end
    %
    %
    % elseif kdim==15
    %
    %     for i=1:kdim-1
    %         s(i)=prod(ns(i+1:kdim));
    %     end
    %
    %     for i=1:kdim
    %         if ns(i)>1
    %             pin(i)=(pmax(i)-pmin(i))/(2*ns(i));
    %             p(i,1:ns(i))=linspace(pmin(i)+pin(i),pmax(i)-pin(i),ns(i));
    %         else
    %             p(i,1)=(pmin(i)+pmax(i))/2;
    %         end
    %     end
    %
    %     for i1=1:ns(1)
    %         for i2=1:ns(2)
    %             for i3=1:ns(3)
    %                 for i4=1:ns(4)
    %                     for i5=1:ns(5)
    %
    %                         for i6=1:ns(6)
    %                             for i7=1:ns(7)
    %                                 for i8=1:ns(8)
    %                                     for i9=1:ns(9)
    %                                         for i10=1:ns(10)
    %
    %                                             for i11=1:ns(11)
    %                                                 for i12=1:ns(12)
    %                                                     for i13=1:ns(13)
    %                                                         for i14=1:ns(14)
    %                                                             for i15=1:ns(15)
    %
    %                                                                 l=(i1-1)*s(1)+(i2-1)*s(2)+(i3-1)*s(3)+(i4-1)*s(4)+(i5-1)*s(5)...
    %                                                                     +(i6-1)*s(6)+(i7-1)*s(7)+(i8-1)*s(8)+(i9-1)*s(9)+(i10-1)*s(10)...
    %                                                                     +(i11-1)*s(11)+(i12-1)*s(12)+(i13-1)*s(13)+(i14-1)*s(14)+i15;
    %
    %                                                                 pgrid(1,l)=p(1,i1);
    %                                                                 pgrid(2,l)=p(2,i2);
    %                                                                 pgrid(3,l)=p(3,i3);
    %                                                                 pgrid(4,l)=p(4,i4);
    %                                                                 pgrid(5,l)=p(5,i5);
    %
    %                                                                 pgrid(6,l)=p(6,i6);
    %                                                                 pgrid(7,l)=p(7,i7);
    %                                                                 pgrid(8,l)=p(8,i8);
    %                                                                 pgrid(9,l)=p(9,i9);
    %                                                                 pgrid(10,l)=p(10,i10);
    %
    %                                                                 pgrid(11,l)=p(11,i11);
    %                                                                 pgrid(12,l)=p(12,i12);
    %                                                                 pgrid(13,l)=p(13,i13);
    %                                                                 pgrid(14,l)=p(14,i14);
    %                                                                 pgrid(15,l)=p(15,i15);
    %
    %                                                             end
    %                                                         end
    %                                                     end
    %                                                 end
    %                                             end
    %
    %                                         end
    %                                     end
    %                                 end
    %                             end
    %                         end
    %
    %                     end
    %                 end
    %             end
    %         end
    %     end
    %
    %
    % end
    
