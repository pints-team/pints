CreateUserEquations  := proc(g,xdim,kdim)

// comments in this procedure are to highlight the differences between MuPad and Maple syntax, not necessarily to explain what is happening
  
local fd, i, si, sname;    

// Write equations  to user_equations.m

begin    
  
fd:=fopen("user_equations.m",Write,Text);                          // Include the word Text, and change WRITE --> Write
if testeq(fd,FAIL) then
    error("Unable to open file for writing, check your WRITEPATH");
end_if;

fprint(Unquoted, fd, "function [fparam,fvec]=user_equations");     // fprintf --> fprint.   There are 2 types: Unquoted (inserts new line automatically) and NoNL (No New Line)

fprint(Unquoted, fd, "");
fprint(Unquoted, fd, "xdim = ", xdim,";");                         // Quotes will give exact text, no quotes will give variable name
fprint(Unquoted, fd, "kdim = ", kdim,";");                         
fprint(Unquoted, fd, "fparam=[xdim,kdim]; \n"); 

for i from 1 to xdim do                                            // for loops start the same, but end with end_for instead of end do:
    si:=int2text(i);                                               // parse(string) is not a command.  If you want to parse a number to a string, use int2text
    sname:=_concat("f",si);                                        // cat(s1,s2,...) --> _concat(s1,s2,...)
    fprint(Unquoted, fd, sname, " = '",g[i],"';");
end_for; 

fprint(Unquoted, fd, "");        // this inserts a new line
for i from 1 to xdim do
    si:=int2text(i); 
    sname:=_concat("f",si); 
    fprint(Unquoted, fd, "fvec(",i,",1:length(",sname,")) = ",sname,";"); 
end_for; 

fprint(Unquoted, fd, "");
fprint(NoNL, fd, "end"); 
fclose(fd);

end_proc;