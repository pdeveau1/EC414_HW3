% EC 414 - HW 3 - Spring 2022
% DP-Means starter code

clear, clc, close all,

%% Generate Gaussian data:
% Add code below:


%% Generate NBA data:
% Add code below:

% HINT: readmatrix might be useful here

%% DP Means method:

% Parameter Initializations
LAMBDA = 0.15;
convergence_threshold = 1;
num_points = length(DATA);
total_indices = [1:num_points];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DP Means - Initializations for algorithm %%%
% cluster count
K = 1;

% sets of points that make up clusters
L = {};
L = [L [1:num_points]];

% Class indicators/labels
Z = ones(1,num_points);

% means
MU = [];
MU = [MU; mean(DATA,1)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initializations for algorithm:
converged = 0;
t = 0;
while (converged == 0)
    t = t + 1;
    fprintf('Current iteration: %d...\n',t)
    
    %% Per Data Point:
    for i = 1:num_points
        
        %% CODE 1 - Calculate distance from current point to all currently existing clusters
        % Write code below here:
        
        %% CODE 2 - Look at how the min distance of the cluster distance list compares to LAMBDA
        % Write code below here:

    end
    
    %% CODE 3 - Form new sets of points (clusters)
    % Write code below here:
    
    %% CODE 4 - Recompute means per cluster
    % Write code below here:
    
    %% CODE 5 - Test for convergence: number of clusters doesn't change and means stay the same %%%
    % Write code below here:
    
    %% CODE 6 - Plot final clusters after convergence 
    % Write code below here:
    
    if (converged)
        %%%%
    end    
end




