CreateUserParameters := proc(cp)

local fd;

begin
  
// Write user defined parameters to user_parameters.m
fd:=fopen("user_parameters.m", Text, Write); 
fprint(Unquoted, fd, "function [cp]=user_parameters \n"); 
fprint(Unquoted, fd, "cp = '",cp,"';\n");
fprint(NoNL, fd, "end"); 
fclose(fd);

end_proc;