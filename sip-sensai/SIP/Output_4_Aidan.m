
function Output_4_Aidan(DirectoryName,JobName)
%
%  ***   Output_4_Aidan(DirectoryName,JobName)   ***
%
% Create a text file of p1 p2 p3 probability from 1 to number of cells
%

% Load mat file created by SENSAI to obtain geometry information
file1=[DirectoryName,'/',JobName,'/SIP_matfile.mat']

% Load mat file created by SIP to obtain recovered input probability distribution
file2=[DirectoryName,'/',JobName,'/SIP_SENSAI_output.mat']
load(file1)
load(file2)
[m1,n1]=size(geom);
[m2,n2]=size(pinp);

% Check number of rows in geom file is the same as the number of rows in
% the file of recovered probabilities
if m1~=m2
    fprintf('We have a problem \n')
    return
end

x=zeros(m1,n1);
x(1:m1,1:n1-1)=geom(1:m1,1:n1-1);
x(1:m1,n1)=pinp(1:m1,n2);

PlotfileName=[DirectoryName,'/',JobName,'/',JobName,'_plotfile.txt']
fid=fopen(PlotfileName,'w');

if n1==3
    for i=1:m1
        fprintf(fid,'%13.6e  %13.6e  %13.6e \n', x(i,1:n1));
    end
elseif n1==4
    for i=1:m1
        fprintf(fid,'%13.6e  %13.6e  %13.6e  %13.6e \n', x(i,1:n1));
    end
end

fclose(fid);

end