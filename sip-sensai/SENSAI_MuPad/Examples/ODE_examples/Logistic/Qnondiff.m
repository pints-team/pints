function qsample=Qnondiff(t,x,p,Qindex,nrsample,obserror)
%
%  ***   qsample=Qnondiff(t,x,p,Qindex,nrsample,obserror)   ***
%
qsample=zeros(1,nrsample);
[mx,nx]=size(x);

qsample(1,1)=x(1,end);
              
end

