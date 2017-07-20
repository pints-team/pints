cp := proc(gparam,cp)

local fd, xdim, kdim, fd, cps, cpx, j, dcdp, sj, sname;

begin

fd:=fopen("cp.m",Text, Write); 

fprint(Unquoted, fd, "function [cparam,dcdp] = cp(x,p,xdim,kdim) \n");
fprint(Unquoted, fd, "dcdp = zeros(kdim+xdim,1); \n");

xdim:=gparam[1];
kdim:=gparam[2];

fd2 := fopen("MuPadRoutines/mm_interface.out",Append);
fprint(Unquoted, fd2, "cp");
fprint(Unquoted, fd2, "cp = ", cp);
fprint(Unquoted, fd2, "xdim = ", xdim);
fprint(Unquoted, fd2, "kdim = ", kdim);

fprint(Unquoted, fd2, "cps = ", cp);

//cpx := text2expr(cp);
//fprint(Unquoted, fd2, "cpx = ", cpx);

fclose(fd2);

fprint(Unquoted, fd, generate::MATLAB(cparam = cp));

for j from 1 to kdim+xdim do:
   dcdp[j]:= diff(cp,p[j]);
end_for:

for j from 1 to kdim+xdim do:
   sname := _concat("dcdp",j);
   xname := text2expr(sname);
   fprint(NoNL, fd, generate::MATLAB(xname = dcdp[j]));
end_for:
 
fprint(Unquoted, fd, ""); 

for j from 1 to kdim+xdim do:
     fprint(Unquoted, fd, "dcdp(",j,") = dcdp",j,";");
end_for:

fprint(Unquoted, fd, "");
fprint(NoNL, fd, "end");

fclose(fd);

end_proc;
