function [qoi,qdim]=user_QoI

qdim = 1;
q1 = 'x[2]';

qoi(1,1:length(q1)) = q1;

end