dgvec_dxvec := proc(gparam,gtotal)

local xdim, kdim, fd, gs, i, si, sj, gx, j, sname, dgdx;

begin

fd:=fopen("dgvec_dxvec.m",Text,Write); 
fprint(Unquoted, fd, "function [dgdx] = dgvec_dxvec(t,x,p) \n");
fprint(Unquoted, fd, "xdim=length(x);");
fprint(Unquoted, fd, "dgdx = zeros(xdim,xdim); \n");

xdim:=gparam[1];
kdim:=gparam[2];

fd2 := fopen("MuPadRoutines/mm_interface.out",Append);
fprint(Unquoted, fd2, "dgvec_dxvec");
fprint(Unquoted, fd2, "gtotal = ",gtotal);
fprint(Unquoted, fd2, "xdim = ", xdim);
fprint(Unquoted, fd2, "kdim = ", kdim);

gs:=stringlib::split(gtotal,";");
fprint(Unquoted, fd2, "gs = ",gs);

for i from 1 to xdim do
  fprint(Unquoted, fd2, "gs[",i,"] = ",gs[i]);
end_for:

for i from 1 to xdim do
  gx[i]:=text2expr(gs[i]):
end_for:

for i from 1 to xdim do:
  fprint(Unquoted, fd2, "gx[",i,"] = ",gx[i]);
end_for:

//for i from 1 to xdim do:
//   for j from 1 to xdim do:
//      fprint(Unquoted, fd2, i,j,", sname = dg",i,"dx",j);
//   end do:
//end do:

fclose(fd2);

for i from 1 to xdim do:
    for j from 1 to xdim do:
         dgdx[i,j]:= diff(gx[i],x[j]);
    end_for:
end_for:

for i from 1 to xdim do:
   for j from 1 to xdim do:
      sname := _concat("dg",i,"dx",j);
      xname := text2expr(sname);
      fprint(NoNL, fd, generate::MATLAB(xname = dgdx[i,j]));
   end_for:
end_for:
 
fprint(Unquoted, fd, "");

for i from 1 to xdim do:
   for j from 1 to xdim do:
      fprint(Unquoted, fd, "dgdx(",i,",",j,") = dg",i,"dx",j,";");
   end_for:
end_for:
 
fprint(Unquoted, fd, "");
fprint(NoNL, fd, "end");

fclose(fd);

end_proc;

