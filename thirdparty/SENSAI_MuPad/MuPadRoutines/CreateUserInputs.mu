CreateUserInputs := proc(DIR,JOB,imap,x0,xdim,p0,kdim,tfinal,solntimes,ntsteps,qtype,stype,NextGen,R0_only)

local fd, i, n, m;

begin

// Write initial conditions and parameter values to user_inputs.m
fd:=fopen("user_inputs.m",Text, Write); 
fprint(Unquoted, fd, "function [DIR,JOB,imap,x0,p0,tfinal,solntimes,ntsteps,qtype,stype,NextGen,R0_only]=user_inputs \n"); 

fprint(Unquoted, fd, "DIR = '",DIR,"';");
fprint(Unquoted, fd, "JOB = '",JOB,"';");

fprint(Unquoted, fd, ""); 
fprint(Unquoted, fd, "imap = ",imap,";");

fprint(Unquoted, fd, ""); 
for i from 1 to xdim do  
  fprint(Unquoted, fd, "x0(",i,",1) = ",x0[i],";"); 
end_for; 

fprint(Unquoted, fd, ""); 
for i from 1 to kdim do  
  fprint(Unquoted, fd, "p0(",i,",1) = ",p0[i],";"); 
end_for; 

fprint(Unquoted, fd, ""); 
fprint(Unquoted, fd, "tfinal = ",tfinal,";"); 
fprint(Unquoted, fd, "solntimes = ",solntimes,";"); 
fprint(Unquoted, fd, "ntsteps = ",ntsteps,";"); 

fprint(Unquoted, fd, "");
fprint(Unquoted, fd, "qtype = ",qtype,";");
fprint(Unquoted, fd, "stype = ",stype,";");

fprint(Unquoted, fd, "");
n := linalg::matdim(NextGen)[1];
for i from 1 to n do
   fprint(Unquoted, fd, "NextGen(",i,",1) = ",NextGen[i],";");
end_for:
fprint(Unquoted, fd, "R0_only = ",R0_only,";");

fprint(Unquoted, fd, ""); 
fprint(NoNL, fd, "end"); 

fclose(fd);

end_proc;
