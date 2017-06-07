function [qoi,qdim]=user_QoI

qdim = 3;
q1 = 'x[4]/(x[2] + x[3] + x[4] + x[5])';
q2 = 'x[3]/(x[2] + x[3] + x[4] + x[5])';
q3 = '(t*x[5])/(x[2] + x[3] + x[4] + x[5])';

qoi(1,1:length(q1)) = q1;
qoi(2,1:length(q2)) = q2;
qoi(3,1:length(q3)) = q3;

end