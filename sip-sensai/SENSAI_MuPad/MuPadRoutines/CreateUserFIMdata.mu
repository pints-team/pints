CreateUserFIMdata := proc(eFIM,qFIM,Fdim,Fp,iFtimes,pest,sig)

local fd, i;

begin

// Write FIM data to user_FIMdata.m
fd:=fopen("user_FIMdata.m",Text, Write); 
fprint(Unquoted, fd, "function [eFIM,qFIM,Fdim,Fp,iFtimes,pest,sig]=user_FIMdata"); 

fprint(Unquoted, fd, ""); 
fprint(Unquoted, fd, "eFIM = ",eFIM,";"); 
fprint(Unquoted, fd, "qFIM = ",qFIM,";"); 

fprint(Unquoted, fd, ""); 
fprint(Unquoted, fd, "Fdim = ",Fdim,";");
for i from 1 to Fdim do;
    fprint(Unquoted, fd, "Fp(",i,") = ",Fp[i],";"); 
end_for; 
fprint(Unquoted, fd, "iFtimes = ",iFtimes,";");

fprint(Unquoted, fd, ""); 
fprint(Unquoted, fd, "pest = ",pest,";"); 
fprint(Unquoted, fd, "sig = ",sig,";"); 

fprint(Unquoted, fd, ""); 
fprint(NoNL, fd, "end"); 

fclose(fd);

end_proc;

