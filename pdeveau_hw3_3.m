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

% data1 =  mu1' + randn(points,2)*sigma1;
% data2 =  mu2' + randn(points,2)*sigma2;
% data3 =  mu3' + randn(points,2)*sigma3;
data1 = mvnrnd(mu1, sigma1, points);
data2 = mvnrnd(mu2, sigma2, points);
data3 = mvnrnd(mu3, sigma3, points);

mu = [mu1'; mu2'; mu3'];


%% (d)
DATA = [data1 ; data2; data3];

size = length(DATA);
WCSS_d = zeros(9,1);
for(K = 2:10)
    for(i = 1:10)
        index = randi([1,size],1,K); %generate index for three random points from data set for mu
        while(length(unique(index)) ~= K) %if indices are not all unique keep generating indices
            index = randi([1,size],1,K);
        end
        k(K-1).d(i).MU_init = DATA(index, :); %get initial mu values from dataset using randomly generated indices
        [k(K-1).d(i).MU_current, k(K-1).d(i).labels, k(K-1).d(i).WCSS] = k_means(K, k(K-1).d(i).MU_init, DATA);
        %graph_clusters(DATA, d(i).labels, d(i).MU_current);
        k(K-1).WCSS(i) = k(K-1).d(i).WCSS;
    end
    sort_WCSS = sort(k(K-1).WCSS); %sort WCSS from smallest value to largest value
    WCSS_d(K-1) = sort_WCSS(1);
end
figure()
plot(2:10, WCSS_d)
xlabel('k value')
ylabel('WCSS')
title('WCSS change with k-value')

%% 3.3
%f(k, λ) = WCSSk−means + λk, where WCSSk−means
%λ ∈ {15, 20, 25, 30}, plot f(k, λ) as a function of k for k ∈ Krange
K = [2:10];
lambda = [15 20 25 30];
figure()
for i = 1:length(lambda);
    fk = WCSS_d + lambda(i).*K';
    plot(K',fk)
    hold on
end
xlabel('k value')
ylabel('WCSS-means + λk')
legend('15','20','25','30')
title(legend, 'λ')
title('k-means WCSS + penalty λk')
hold off


%% Functions


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
    label = label(1); %if there is a tie in the distance choose the first label
end

%creates a scatter plot of the clusters created and their mu values given the data points, label of those points, and mu values
function graph_clusters(data, labels, mu)
    data1 = data(find(labels == 1), :);
    data2 = data(find(labels == 2), :);
    data3 = data(find(labels == 3), :);
    
    figure()
    scatter(data1(:,1), data1(:,2), 'r'); %plot points with label 1
    hold on
    scatter(data2(:,1), data2(:,2), 'g'); %plot points with label 2
    scatter(data3(:,1), data3(:,2), 'b'); %plot points with label 3
    scatter(mu(:,1),mu(:,2),'black*'); %plot mu values
    xlabel('x1')
    ylabel('x2')
    title('Clusters produced by k-means')
    legend('1','2','3','mu');
    title(legend,'Class');
    hold off
end

%takes K value, initial values of mu, and the data set
%returns the final mu values after k_means has converged
%returns the labels of each data point
%returns the WCSS after convergence
function [MU_current, labels, WCSS] = k_means(K, MU_init, DATA)
    MU_previous = MU_init;
    MU_current = MU_init;
    
    % initializations
    
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
            if(~isempty(label)) %check if cluster is not empty
                MU_current(i,:) = mean(label); %find mean of data points with label and update current mu
            end
        end
        %% CODE 4 - Check for convergence 
        % Write code below here:
        
        %check if the change between the previous mu and the current mu is smaller than the threshhold, if it is k-means has converged
        if (abs(MU_previous - MU_current) < convergence_threshold)
            converged=1;
        end  
        %% CODE 5 - Plot clustering results if converged:
        % Write code below here:
        if (converged == 1)
            fprintf('\nConverged.\n')
          
            %% If converged, get WCSS metric
            % Add code below
            WCSS = 0;
            for(i = 1:K) %loop through each label
                dist = euclidean_distance(MU_current(i,:), DATA(find(labels == i),:)); %find all data points with that label
                WCSS = WCSS + sum(dist.^2);
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end