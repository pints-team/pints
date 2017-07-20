function BET_datafiles(output_directory,qdim,kdim,pref,qref,...
                       nobs,nvol,psample,qsample)
%
%   Detailed explanation goes here


fprintf('Saving pref.txt \n')
fidp=fopen([output_directory,'/pref.txt'],'wt');
for k=1:kdim
    fprintf(fidp,'  %13.6e',pref(k));
end
fprintf(fidp,'\n');
fclose(fidp);

fprintf('Saving qref.txt \n')
fidq=fopen([output_directory,'/qref.txt'],'wt');
for j=1:qdim
    fprintf(fidq,'  %13.6e',qref(j));
end
fprintf(fidq,'\n');
fclose(fidq);

fprintf('Saving psamples.txt \n')
fidp=fopen([output_directory,'/psamples.txt'],'wt');
for i=1:nvol
    for k=1:kdim
        fprintf(fidp,'  %13.6e',psample(k,i));
    end
    fprintf(fidp,'\n');
end
fclose(fidp);

Qdim=qdim*nobs;

for is=1:nvol
    for it=1:nobs
        k=qdim*(it-1)+1;
        Qvals(is,k:k+qdim-1)=qsample(1:qdim,it,is);
    end
end

fprintf('Saving qsamples.txt \n')
fidq=fopen([output_directory,'/qsamples.txt'],'wt');
for i=1:nvol
    for j=1:qdim
        fprintf(fidq,'  %13.6e',Qvals(i,j));
    end
    fprintf(fidq,'\n');
end
fclose(fidq);



end

