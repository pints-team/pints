function [qoi,qdim]=user_QoI

qdim = 1;
q1 = 'x[1]';

qoi(1,1:length(q1)) = q1;

end