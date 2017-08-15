<?
#
# main.m :: This will become the driver file
#
# This file is part of Myokit
#  Copyright 2011-2017 Maastricht University
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
# Authors:
#  Michael Clerx
#
?>%
% Driver file for <?= model.name() ?>
%

% Clear everything up and begin
close all
clear all

% Initial values
<?
for eq in model.inits():
    print(e(eq) + ';')
?>

% Vector for initial values
y0 = [<?= ' '.join([v(var) for var in model.states()]) ?>]';

% Constants
c = constants();

% Pacing
<?
pac = (1000, 2, 50)
?>% Pacing cycle length [ms]
c.pcl = <?= pac[0] ?>;
% Stimulus duration
c.stim_duration = <?= pac[1] ?>;
% Time the first stimulus is given [ms]
c.stim_offset = <?= pac[2] ?>;  

% Get ode solver options
if exist('OCTAVE_VERSION')
    options = odeset( ...
        'RelTol', 1e-6, ...
        'AbsTol', 1e-6, ...
        'InitialStep', 1e-6, ...
        'MaxStep', 10, ...
        'NewtonTol', 1e-6, ...
        'MaxNewtonIterations', 7);
    func = @ode5r;
else
    options=[];
    func = @ode15s;
end

% Simulate!
T = [];
Y = [];
nBeats = 2;
for iBeat = [1:nBeats]
    t1 = (iBeat - 1) * c.pcl;
    t2 = t1 + c.stim_offset;
    fprintf('Beat %d, t = %f\n', iBeat, t1)
    [t, y] = func(@model_wrapper, [t1, t2], y0, options, c);
    y0 = y(size(y,1),:);
    T = [T; t];
    Y = [Y; y];   
    t1 = t2;
    t2 = t1 + c.stim_duration;
    fprintf('Beat %d, t = %f :: Stimulus!\n', iBeat, t1)
    [t, y] = func(@model_wrapper, [t1, t2], y0, options, c);
    y0 = y(size(y,1),:);
    T = [T; t];
    Y = [Y; y];
    t1 = t2;
    t2 = iBeat * c.pcl;
    fprintf('Beat %d, t = %f :: Stimulus over\n', iBeat, t1)
    [t, y] = func(@model_wrapper, [t1, t2], y0, options, c);
    y0 = y(size(y,1),:);
    T = [T; t];
    Y = [Y; y];    
end

% Get results from state vector
<?
i = 0
for var in model.states():
    i += 1
    print(v(var) + ' = Y(:, ' + str(i) + ');')
?>
t = T;

% Show result
figure
plot(t, <?= v(model.states().next()) ?>)
