CreateUserBifndata := proc(ilambda,imu,nu,ds,nstep)

local fd, i;

begin

// Write FIM data to user_bifndata.m
fd:=fopen("user_bifndata.m",Text, Write); 
fprint(Unquoted, fd, "function [ilambda,imu,nu,ds,nstep]=user_bifndata"); 

fprint(Unquoted, fd, ""); 
fprint(Unquoted, fd, "ilambda = ",ilambda,";"); 
fprint(Unquoted, fd, "imu = ",imu,";");
fprint(Unquoted, fd, "nu = ",nu,";"); 
fprint(Unquoted, fd, "ds = ",ds,";");
fprint(Unquoted, fd, "nstep = ",nstep,";"); 

fprint(Unquoted, fd, ""); 
fprint(NoNL, fd, "end"); 

fclose(fd);

end_proc;

