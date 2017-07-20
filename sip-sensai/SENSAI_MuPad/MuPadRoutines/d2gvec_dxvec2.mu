d2gvec_dxvec2 := proc(gparam,gtotal)

local xdim, kdim, fd, gs, i, si, sj, gx, j, sname, dgdx;

begin

fd:=fopen("d2gvec_dxvec2.m",Text,Write); 
fprint(Unquoted, fd, "function [d2gdx2] = dgvec_dxvec2(x,p) \n");
fprint(Unquoted, fd, "xdim=length(x);");
fprint(Unquoted, fd, "kdim=length(p);");
fprint(Unquoted, fd, "d2gdx2 = zeros(xdim,xdim,xdim); \n");

xdim:=gparam[1];
kdim:=gparam[2];

fd2 := fopen("MuPadRoutines/mm_interface.out",Append);
fprint(Unquoted, fd2, "d2gvec_dxvec2");
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
        for k from 1 to xdim do:
            d2gdx2[i,j,k]:= diff(dgdx[i,j],x[k]);
        end_for:
    end_for:
end_for:

for i from 1 to xdim do:
    for j from 1 to xdim do:
        for k from 1 to xdim do:
            sname := _concat("d2g",i,"dx",j,"dx",k);
            xname := text2expr(sname);
            fprint(NoNL, fd, generate::MATLAB(xname = d2gdx2[i,j,k]));
        end_for:
    end_for:
end_for:
 
fprint(Unquoted, fd, "");

for i from 1 to xdim do:
    for j from 1 to xdim do:
        for k from 1 to xdim do:
            fprint(Unquoted, fd, "d2gdx2(",i,",",j,",",k,") = d2g",i,"dx",j,"dx",k,";");
        end_for:
    end_for:
end_for:
 
fprint(Unquoted, fd, "");
fprint(NoNL, fd, "end");

fclose(fd);

end_proc;

