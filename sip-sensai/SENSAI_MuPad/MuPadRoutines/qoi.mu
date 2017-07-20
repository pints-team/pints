qoi := proc(gparam,qtotal)

local xdim, kdim, qdim, qx, dqdx, dqdp, sname, si, sk; 

begin

fd:=fopen("qoi.m",Text,Write); 

fprint(Unquoted, fd, "function [q,dqdx,dqdp] = qoi(t,x,p,xdim,kdim,qdim) \n");
fprint(Unquoted, fd, "q=zeros(qdim,1);");
fprint(Unquoted, fd, "dqdx=zeros(qdim,xdim);");
fprint(Unquoted, fd, "dqdp=zeros(qdim,kdim+xdim); \n");

xdim:=gparam[1];
kdim:=gparam[2];
qdim:=gparam[3];

fd2 := fopen("MuPadRoutines/mm_interface.out",Append); 
fprint(Unquoted, fd2, "qoi");
fprint(Unquoted, fd2, "qtotal = ", qtotal);
fprint(Unquoted, fd2, "xdim = ", xdim);
fprint(Unquoted, fd2, "kdim = ", kdim);
fprint(Unquoted, fd2, "qdim = ", qdim);

qs:=stringlib::split(qtotal,";");
fprint(Unquoted, fd2, "qs = ",qs);

for i from 1 to qdim do:
  fprint(Unquoted, fd2, "qs[",i,"] = ",qs[i]);
end_for:

for i from 1 to qdim do:
  qx[i]:= text2expr(qs[i])         // this is the crux, much like parse() in Maple syntax
end_for:

for i from 1 to qdim do:
  fprint(Unquoted, fd2, "qx[",i,"] = ", qx[i]);
end_for:

fclose(fd2);

// Print qoi
for i from 1 to qdim do:
   sname := _concat("q",i);   // unfortunately, q(i) is an invalide LHS assignment for the MATLAB code generation, so this roundabout way must still be done.
   xname := text2expr(sname);  // must be an expression and not a string
   fprint(NoNL, fd, generate::MATLAB(xname = qx[i]));   // this takes the MuPad syntax and makes it MATLAB syntax.  Both the LHS and RHS must be expressions!
end_for:

fprint(Unquoted, fd, "");

for i from 1 to qdim do:
    fprint(Unquoted, fd, "q(",i,") = q",i,";");
end_for:

fprint(Unquoted, fd, "");

// Compute derivatives wrt the variables x
for i from 1 to qdim do:
    for j from 1 to xdim do:
         dqdx[i,j]:= diff(qx[i],x[j]);
    end_for:
end_for:

for i from 1 to qdim do:
   for j from 1 to xdim do:
      sname := _concat("dq",i,"dx",j);
      xname := text2expr(sname);
      fprint(NoNL, fd, generate::MATLAB(xname = dqdx[i,j]));
   end_for:
end_for:
 
fprint(Unquoted, fd, "");

for i from 1 to qdim do:
   for j from 1 to xdim do:
      fprint(Unquoted, fd, "dqdx(",i,",",j,") = dq",i,"dx",j,";");
   end_for:
end_for:
 
fprint(Unquoted, fd, "");


// Compute derivatives wrt the parameters p
for i from 1 to qdim do:
    for j from 1 to kdim do:
         dqdp[i,j]:= diff(qx[i],p[j]);
    end_for:
end_for:

for i from 1 to qdim do:
   for j from 1 to kdim do:
      sname := _concat("dq",i,"dp",j);
      xname := text2expr(sname);
      fprint(NoNL, fd, generate::MATLAB(xname=dqdp[i,j]));
   end_for:
end_for:

fprint(Unquoted, fd, "");  

for i from 1 to qdim do:
   for j from 1 to kdim do:
      fprint(Unquoted, fd, "dqdp(",i,",",j,") = dq",i,"dp",j,";");
   end_for:
end_for:

fprint(Unquoted, fd, "");
fprint(NoNL, fd, "end");

fclose(fd);

end_proc;
