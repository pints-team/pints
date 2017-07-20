function [qoi,qdim]=user_QoI

qdim = 3;
q1 = 'x[1] + x[2] + x[3] + x[4]';
q2 = 'x[2]/p[1]';
q3 = 'x[3]/p[1]';

qoi(1,1:length(q1)) = q1;
qoi(2,1:length(q2)) = q2;
qoi(3,1:length(q3)) = q3;

end