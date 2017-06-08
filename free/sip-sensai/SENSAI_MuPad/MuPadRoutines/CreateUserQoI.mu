CreateUserQoI := proc(qoi,qdim)

local fd, i, si, sname;

//with(CodeGeneration):
//with(StringTools):

begin

// Write user defined parameters to user_QoI.m
fd:=fopen("user_QoI.m",Text,Write); 
fprint(Unquoted, fd, "function [qoi,qdim]=user_QoI"); 

fprint(Unquoted, fd, ""); 
fprint(Unquoted, fd, "qdim = ", qdim,";");
for i from 1 to qdim do                                            // for loops start the same, but end with end_for instead of end do:
    si:=int2text(i);                                               // parse(string) is not a command.  If you want to parse a number to a string, use int2text
    sname:=_concat("q",si);                                        // cat(s1,s2,...) --> _concat(s1,s2,...)
    fprint(Unquoted, fd, sname, " = '",qoi[i],"';");
end_for; 

fprint(Unquoted, fd, "");        // this inserts a new line
for i from 1 to qdim do
    si:=int2text(i); 
    sname:=_concat("q",si); 
    fprint(Unquoted, fd, "qoi(",i,",1:length(",sname,")) = ",sname,";"); 
end_for; 

fprint(Unquoted, fd, "");
fprint(NoNL, fd, "end"); 

fclose(fd);

end_proc;