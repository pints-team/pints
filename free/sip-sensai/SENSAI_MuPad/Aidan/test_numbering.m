    

% kdim = 6
% ns=[1 2 2 2 4 4]
% prod(ns)
%     
%     cnt=0;
%     for i=1:6
%         for j=1:ns(i)
%             cnt=cnt+1;
%             p(i,j)=cnt;
%         end
%     end
%     p
%    
%     s(6)=1;
%     for i=5:-1:1
%         s(i)=ns(i+1)*s(i+1);
%     end
%     s
%     
% %     for i=1:6
% %         pin(i)=(pmax(i)-pmin(i))/(2*ns(i));
% %         p(i,:)=linspace(pmin(i)+pin(i),pmax(i)-pin(i),ns(i))
% %     end
%     for i1=1:ns(1)
%         for i2=1:ns(2)
%             for i3=1:ns(3)
%                 for i4=1:ns(4)
%                     for i5=1:ns(5)
%                         for i6=1:ns(6)
%                             l=(i1-1)*s(1)+(i2-1)*s(2)+(i3-1)*s(3)+(i4-1)*s(4)+(i5-1)*s(5)+ i6;
%                             pgrid(1,l)=p(1,i1);
%                             pgrid(2,l)=p(2,i2);
%                             pgrid(3,l)=p(3,i3);
%                             pgrid(4,l)=p(4,i4);
%                             pgrid(5,l)=p(5,i5);
%                             pgrid(6,l)=p(6,i6);
%                         end
%                     end
%                 end
%             end
%         end
%     end
%     pgrid
%     pause
%
%


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% kdim = 15
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
% kdim = 15
ns=[1 2 2 2 1   1 2 2 3 1    1 1 2 1 1 ]
prod(ns)

kdim=length(ns)
    
%     cnt=0;
%     for i=1:kdim
%         for j=1:ns(i)
%             cnt=cnt+1;
%             p(i,j)=cnt;
%         end
%     end
%     p
   
    s(kdim)=1;
    for i=1:kdim-1
        s(i)=prod(ns(i+1:kdim));
    end
    s
        
    pmin=[1 2 3 4 5   6 7 8 9 10  11 12 13 14 15];
    pmax=20+pmin;
    
    for i=1:15
        if ns(i)>1
            pin(i)=(pmax(i)-pmin(i))/(2*ns(i));
            p(i,1:ns(i))=linspace(pmin(i)+pin(i),pmax(i)-pin(i),ns(i));
        else
            p(i,1)=(pmin(i)+pmax(i))/2;
        end
    end
    p
    
    for i1=1:ns(1)
        for i2=1:ns(2)
        for i3=1:ns(3)
        for i4=1:ns(4)
        for i5=1:ns(5)
            
        for i6=1:ns(6)
        for i7=1:ns(7)
        for i8=1:ns(8)
        for i9=1:ns(9)
        for i10=1:ns(10)
                    
        for i11=1:ns(11)
        for i12=1:ns(12)
        for i13=1:ns(13)
        for i14=1:ns(14)
        for i15=1:ns(15)
                   
            l=(i1-1)*s(1)+(i2-1)*s(2)+(i3-1)*s(3)+(i4-1)*s(4)+(i5-1)*s(5)...
              +(i6-1)*s(6)+(i7-1)*s(7)+(i8-1)*s(8)+(i9-1)*s(9)+(i10-1)*s(10)...
              +(i11-1)*s(11)+(i12-1)*s(12)+(i13-1)*s(13)+(i14-1)*s(14)+i15;
           
            pgrid(1,l)=p(1,i1);
            pgrid(2,l)=p(2,i2);
            pgrid(3,l)=p(3,i3);
            pgrid(4,l)=p(4,i4);
            pgrid(5,l)=p(5,i5);
                                      
            pgrid(6,l)=p(6,i6);
            pgrid(7,l)=p(7,i7);
            pgrid(8,l)=p(8,i8);
            pgrid(9,l)=p(9,i9);
            pgrid(10,l)=p(10,i10);
                                      
            pgrid(11,l)=p(11,i11);
            pgrid(12,l)=p(12,i12);
            pgrid(13,l)=p(13,i13);
            pgrid(14,l)=p(14,i14);
            pgrid(15,l)=p(15,i15);
                            
        end
        end
        end
        end
        end
        
        end
        end
        end
        end
        end
        
        end
        end
        end
        end
    end
    pgrid

    