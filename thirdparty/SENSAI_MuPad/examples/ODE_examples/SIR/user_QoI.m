function [qoi,qdim]=user_QoI

qdim = 2;
q1 = 'x[2]/(x[1] + x[2] + x[3])';
q2 = 'x[2]';

qoi(1,1:length(q1)) = q1;
qoi(2,1:length(q2)) = q2;

end