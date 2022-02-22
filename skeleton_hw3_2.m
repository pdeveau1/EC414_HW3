% EC 414 - HW 3 - Spring 2022
% K-Means starter code

clear, clc, close all;

%% Generate Gaussian data:
% Add code below:
mu1 = [2,2]';
mu2 = [-2,2]';
mu3 = [0, -3.25]';

I2 = [1 0; 0 1];

sigma1 = 0.02 * I2;
sigma2 = 0.05 * I2;
sigma3 = 0.07 * I2;

points = 50;

data1 =  mu1' + randn(points,2)*sigma1;
data2 =  mu2' + randn(points,2)*sigma2;
data3 =  mu3' + randn(points,2)*sigma3;
% data1 = mvnrnd(mu1, sigma1, points);
% data2 = mvnrnd(mu2, sigma2, points);
% data3 = mvnrnd(mu3, sigma3, points);

figure()
scatter(data1(:,1), data1(:,2), 'r');
hold on
scatter(data2(:,1), data2(:,2), 'g');
scatter(data3(:,1), data3(:,2), 'b');
title('Gaussian data points')
legend('1','2','3');
hold off
% %% Generate NBA data:
% % Add code below:
% 
% % HINT: readmatrix might be useful here
% 
% % Problem 3.2(f): Generate Concentric Rings Dataset using
% % sample_circle.m provided to you in the HW 3 folder on Blackboard.
% 
%% K-Means implementation
% Add code below

K = 3;
MU_init = [3 3; -4 -1; 2 -4];
%MU_init = [-0.14 2.61; 3.15 -0.84; -3.28 -1.58];

MU_previous = MU_init;
MU_current = MU_init;

% initializations
DATA = [data1 ; data2; data3];
labels = ones(length(DATA),1);
converged = 0;
iteration = 0;
convergence_threshold = 0.025;

%stop if the derived cluster means become stationary
while (converged==0)
    iteration = iteration + 1;
    fprintf('Iteration: %d\n',iteration)

    %% CODE - Assignment Step - Assign each data observation to the cluster with the nearest mean:
    % Write code below here:
    for(i = 1:length(DATA))
        dist = euclidean_distance(DATA(i,:),MU_current); %find euclidean distance of each data point to each mu
        labels(i) = closest_label(dist, MU_current); %find the label of the closest mu       
    end
    %% CODE - Mean Updating - Update the cluster means
    % Write code below here:
    MU_previous = MU_current;
    
    for(i = 1:K) %loop through each label
        label = DATA(find(labels == i),:); %find all data points with that label 
        MU_current(i,:) = mean(label); %find mean of data points with label and update current mu
    end

    %% CODE 4 - Check for convergence 
    % Write code below here:
    
    %check if the change between the previous mu and the current mu is smaller than the threshhold, if it is k-means has converged
    %check if no points map to one of mu, if so k-means has converged
    if (sum(sum(abs(MU_previous - MU_current) < convergence_threshold)) || sum(sum(isnan(MU_current))))
        converged=1;
    end
    
    %% CODE 5 - Plot clustering results if converged:
    % Write code below here:
    if (converged == 1)
        fprintf('\nConverged.\n')
        
        
       
        
        %% If converged, get WCSS metric
        % Add code below
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

data1 = DATA(find(labels == 1), :);
data2 = DATA(find(labels == 2), :);
data3 = DATA(find(labels == 3), :);

figure()
scatter(data1(:,1), data1(:,2), 'r');
hold on
scatter(data2(:,1), data2(:,2), 'g');
scatter(data3(:,1), data3(:,2), 'b');
scatter(MU_current(:,1),MU_current(:,2),'black*')
title('Clusters produced by k-means')
legend('1','2','3');
hold off
% 
% 
% 
%returns euclidean distance of a singular point to each point in a matrix
function distance = euclidean_distance(point, points)
    distance = zeros(length(points),1); %initialize distance vector
    [r,c] = size(points);
    for (i = 1:r) % loop through each point of the matrix of points
        distance(i,1) = sqrt((point(1,1)-points(i,1))^2+(point(1,2)-points(i,2))^2); %distance between two points
    end
end

%returns the label of the closest point given a distance vector and matrix of points
function label = closest_label(distance, points)
    sort_dist = sort(distance); %sort the distance vector from smalllest to largest
    close_point = points(find(sort_dist(1) == distance), :); %find the index of the smallest distance in the distance vector and find point that gives that distance
    [tf, label] = ismember(close_point, points, 'rows'); %find row index of the closest mu, correlates to label
end