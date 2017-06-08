function [output_directory]=Make_Output_Directory(output_source,ExampleName,JobName,otype,ofname,Qindex,qdim)
%  ***   [output_directory]=Make_Output_Directory(output_source,ExampleName,JobName,otype,ofname,Qindex,qdim)  ***
%
% Create output directory
%

output_directory=[output_source,'/',ExampleName,'/',JobName];
Qindex_string=num2str(Qindex(1));
for idim=2:qdim
    Qindex_string=[Qindex_string,num2str(Qindex(idim))];
end
output_directory=[output_directory,'_',Qindex_string];

%  When reading output probability distribution from a file
if max(otype)==3
    output_directory=[output_directory,'_',ofname];
end
%
fprintf(['Output directory = ', output_directory, '\n']) 
mkdir(output_directory)

end


