function [nangle,ang1,ang2,ang3]=solve_active_subspace_angles(nt,t,Mdim,Sdim,Fdim,V)
%
%  ***   [nangle,ang1,ang2,ang3]=solve_active_subspace_angles(nt,t,Mdim,Sdim,Fdim,V)   ***
%
%
%  Purpose
%  -------
%    Calculates the active subspace using sensitivities or elasticities of 
%    variables or QoIs
%
%  Variables
%  ---------
% Determine whether to construct the FIM from sensitivities or elasticities
% and whether to use primitive variables or quantities of interest

% Fdim = number of parameters to use to construct FIM
%
%  Calls
%  -----
%
%

ang1=zeros(1,nt);
ang2=zeros(2,nt);
ang3=zeros(3,nt);

%
% Calculate angles between active subspaces
nangle=min(3,Fdim);
fprintf('nangle = %3i \n', nangle)

for idim=1:max(3,Fdim)
    V10=V(1:Fdim,1,2);
    if nangle >= 2
        V20=V(1:Fdim,1:2,2);
    end
    if nangle >= 3
        V30=V(1:Fdim,1:3,2);
    end
end

for it=3:nt
    %     if it==1
    %         itm1=1;
    %     else
    %         itm1=it-1;
    %     end
    
    V1=V(1:Fdim,1,it);
    MM=V10'*V1;
    [Uang,Sang,Vang]=svd(MM);
    ang1(1,it)=acos(diag(Sang));
    %     if ~isreal(ang1(1,it))
    %         fprintf('t = %13.4e, S1 = %13.4e, angle = %13.4e + i*%13.4e \n',...
    %             t(it), S, real(ang1(k,it)), imag(ang1(k,it)))
    %     end
    
    if nangle >= 2
        V2=V(1:Fdim,1:2,it);
        MM=V20'*V2;
        [Uang,Sang,Vang]=svd(MM);
        for k=1:2
            if Sang(k,k) > 1
                Sang(k,k)=1;
            end
            if Sang(k,k) < 0
                Sang(k,k)=0;
            end
        end
        ang2(1:2,it)=acos(diag(Sang));
        for k=1:2
            %             if ~isreal(ang2(k,it))
            %                 fprintf('t = %13.4e, S2(%2i) = %13.4e, angle = %13.4e + i*%13.4e  \n',...
            %                     t(it), k, S(k), real(ang2(k,it)), imag(ang2(k,it)))
            %             end
        end
    end
    
    if nangle >= 3
        V3=V(1:Fdim,1:3,it);
        MM=V30'*V3;
        [Uang,Sang,Vang]=svd(MM);
        for k=1:3
            if Sang(k,k) > 1
                Sang(k,k)=1;
            end
            if Sang(k,k) < 0
                Sang(k,k)=0;
            end
        end
        ang3(1:3,it)=acos(diag(Sang));
        for k=1:3
            %             if ~isreal(ang3(k,it))
            %                 fprintf('t = %13.4e, S3(%2i) = %13.4e, angle = %13.4e + i*%13.4e \n',...
            %                     t(it), k, S(k), real(ang3(k,it)), imag(ang3(k,it)))
            %             end
        end
    end
    
end

end
