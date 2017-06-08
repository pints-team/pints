function [Qindex,fmin,fmax,fnedge,otype,ofname,omean,osigma,pnedge] = SIP_SENSAI_input;
%  ***   [Qindex,fmin,fmax,fnedge,otype,ofname,omean,osigma,pnedge] = SIP_SENSAI_input;  ***
%
% Input
% -----
%  Qindex = indices of outputs used to recover parameters
%  fmin   = minimum values of output quantities 
%  fmax   = maximum values of output quantities
%  fnedge = number of bins for output quantities
%  otype  = type of output probability distribution
%            0 -> uniform distribution
%            1 -> normal distribution
%  omean  = mean of output probability distribution
%  osigma = range or standard deviation of output probability distribution
%  pnedge = number of bins for recovered input parameters
%
%  An option exists to read in a file containing manufactured outputs
%


%Qindex=[1];
%fmin=[1.3];
%fmax=[2.5];
%fnedge=[21];
%otype=[1];
%omean=[1.7];
%osigma=[0.05*1.7];
%pnedge=[21; 21; 21];
%ofname='ttt';

% t=3  -> x=0.78454,  fmin=0.5, fmax=1.1, omean=[0.785];    osigma=[0.05*0.785];
% t=10 -> x=15.10363, fmin=11,  fmax=18,  omean=[15.10363]; osigma=[0.05*15.10363]
% t=23 -> x=17.49968, min=15, max=20


% Hodgkin Huxley ABC example
% ==========================
% Qindex=[6];
% fmin=[0];
% fmax=[1.3];
% fnedge=[21];
% otype=[0];
% lower=[0]; 
% upper=[0.075];
% omean =[(lower+upper)/2];
% osigma=[(upper-lower)/2];
% pnedge=[21; 21; 21];
% ofname='ttt';

% Hodgkin Huxley example
% ======================
Qindex=[1];
fmin=[1.3];
fmax=[2.6];
fnedge=[21];
otype=[1];
omean=[1.65];
osigma=[0.0165];
pnedge=[21; 21; 21];
ofname='ttt';

end


% 0.5 TO 2.0 OMEGA VALUES

% Qindex=[1;    2;    3;   4;    5];
% fmin=[1.3;  3.5;   90;  -8;  -87];
% fmax=[2.6;  4.7;  500;  39;  -80];


% LARGE OMEGA VALUES

% Qindex=[1; 2; 3];
% fmin=[1.3;  3.5;  150];
% fmax=[2.4;  4.4;  425];
% fnedge=[21; 21; 21];
% otype=[1; 1; 1];
% omean =[1.65;   3.65;     317];
% osigma=[0.0165; 0.0365/2; 3.17];
% pnedge=[21; 21; 21];

% Qindex=[3; 4; 5];
% fmin=[150;   9;  -86];
% fmax=[425;  38;  -80];
% fnedge=[21; 21; 21];
% otype=[1; 1; 1];
% omean= [317;   31.0;  -84.1];
% osigma=[3.17;  0.31;   0.841/3];
% pnedge=[21; 21; 21];


% Qindex=[6];
% fmin=[0];
% fmax=[0.75];
% fnedge=[21];
% otype=[0];
% lower=[0]; 
% upper=[0.075];
% omean =[(lower+upper)/2];
% osigma=[(upper-lower)/2];
% pnedge=[21; 21; 21];
% ofname='ttt';



% MEDIUM OMEGA VALUES
% Qindex=[1; 2; 3];
% fmin=[1.3; 3.5; 220];
% fmax=[2.1; 4.1; 420];
% fnedge=[21; 21; 21];
% otype=[1; 1; 1];
% omean =[1.65;   3.65;     317];
% osigma=[0.0165; 0.0365/2; 3.17];
% pnedge=[21; 21; 21];
%
% Qindex=[3; 4; 5];
% fmin=[220;  20;  -86];
% fmax=[420;  36;  -82];
% fnedge=[21; 21; 21];
% otype=[1; 1; 1];
% omean= [317;   31.0;  -84.1];
% osigma=[3.17;  0.31;   0.841/6];
% pnedge=[21; 21; 21];
