gvec := proc(gparam,gtotal)

//local fd, fd2, xdim, kdim, fd, gs, i, gx, sname;  // no need for local variables: this is default
// All can be done in one step: no need for top,middle,bottom
  
begin
  
fd := fopen("gvec.m",Text, Write);           
fprint(Unquoted, fd, "function [g] = gvec(t,x,p) \n"); 

xdim:=gparam[1];
kdim:=gparam[2];

fd2 := fopen("MuPadRoutines/mm_interface.out",Text,Write);  // you can open multiple files at once!
fprint(Unquoted, fd2, "gvec");
fprint(Unquoted, fd2, "gtotal = ", gtotal);
fprint(Unquoted, fd2, "xdim = ", xdim);
fprint(Unquoted, fd2, "kdim = ", kdim);

gs:=stringlib::split(gtotal,";");
fprint(Unquoted, fd2, "gs = ",gs);

for i from 1 to xdim do:
  fprint(Unquoted, fd2, "gs[",i,"] = ",gs[i]);   // eliminates need for si and sname
end_for:

for i from 1 to xdim do:
  gx[i]:= text2expr(gs[i])         // this is the crux, much like parse() in Maple syntax
end_for:

for i from 1 to xdim do:
  fprint(Unquoted, fd2, "gx[",i,"] = ", gx[i]);
end_for:

//for i from 1 to xdim do:
//   fprint(Unquoted, fd2, i,", sname = g",i);    // don't really need sname printed?
//end_for:

fclose(fd2);

for i from 1 to xdim do:
   sname := _concat("g",i);   // unfortunately, g(i) is an invalide LHS assignment for the MATLAB code generation, so this roundabout way must still be done.
   xname := text2expr(sname);  // must be an expression and not a string
   fprint(NoNL, fd, generate::MATLAB(xname = gx[i]));   // this takes the MuPad syntax and makes it MATLAB syntax.  Both the LHS and RHS must be expressions!
end_for:

fprint(Unquoted, fd, "");

for i from 1 to xdim do:
    fprint(Unquoted, fd, "g(",i,") = g",i,";");
end_for:

fprint(Unquoted, fd, "");
fprint(NoNL, fd, "end");

fclose(fd);

end_proc;