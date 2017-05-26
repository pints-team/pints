dgvec_dparam := proc(gparam,gtotal)

local xdim, kdim, fd, gs, i, gx, j, si, sj, sname, dgdp;

begin

fd:=fopen("dgvec_dparam.m",Text, Write); 
fprint(Unquoted, fd, "function [dgdp] = dgvec_dparam(t,x,p) \n");
fprint(Unquoted, fd, "xdim=length(x);");
fprint(Unquoted, fd, "kdim=length(p);");
fprint(Unquoted, fd, "dgdp = zeros(xdim,kdim+xdim); \n");

xdim:=gparam[1];
kdim:=gparam[2];

fd2 := fopen("MuPadRoutines/mm_interface.out",Append);
fprint(Unquoted, fd2, "dgvec_dparam");
fprint(Unquoted, fd2, "gtotal = ", gtotal);
fprint(Unquoted, fd2, "xdim = ", xdim);
fprint(Unquoted, fd2, "kdim = ", kdim);

gs:=stringlib::split(gtotal,";");
fprint(Unquoted, fd2, "gs = ",gs);

for i from 1 to xdim do:
  fprint(Unquoted, fd2, "gs[",i,"] = ",gs[i]);
end_for:

for i from 1 to xdim do:
  gx[i]:=text2expr(gs[i]);
end_for:

for i from 1 to xdim do:
  fprint(Unquoted, fd2, "gx[",i,"] = ",gx[i]);
end_for:

//for i from 1 to xdim do:
//   for j from 1 to kdim do:
//      fprint(Unquoted, fd2, i,j,", sname = dg",i,"dp",j);
//   end do:
//end do:

fclose(fd2);

for i from 1 to xdim do:
    for j from 1 to kdim do:
         dgdp[i,j]:= diff(gx[i],p[j]);
    end_for:
end_for:

for i from 1 to xdim do:
   for j from 1 to kdim do:
      sname := _concat("dg",i,"dp",j);
      xname := text2expr(sname);
      fprint(NoNL, fd, generate::MATLAB(xname=dgdp[i,j]));
   end_for:
end_for:

fprint(Unquoted, fd, "");  

for i from 1 to xdim do:
   for j from 1 to kdim do:
      fprint(Unquoted, fd, "dgdp(",i,",",j,") = dg",i,"dp",j,";");
   end_for:
end_for:

fprint(Unquoted, fd, "");
fprint(NoNL, fd, "end");

fclose(fd);

end_proc;
