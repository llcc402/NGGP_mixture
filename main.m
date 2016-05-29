%% loading data and preprocessing
clear

data = load('seeds');
data(:,8) = []; % delete labels

[U, S, V] = svd(data);
data = U(:,1:2); % only use the first two important components

S = S(1:7,:);
s = diag(S);
variance_proportion = sum(s(1:2)) / sum(s); % show the variance we captured

% scale data
data = data ./ repmat(std(data), 210, 1);

% scatter(data(:,1), data(:,2), [], [ones(1,70), ones(1,70)*2, ones(1,70)*3])

%% parameter settings

% parameters in generalized Gamma process
alpha = 10;
sigma = 0.95;
tau = 0.5;

% parameters in sampling
maxIter = 500;
pace = 0.1; % used in update u

% parameters in likelihood
lambda_1 = 3; % the precision

% parameters in base measure
lambda_0 = 1;

%% these variables are to be returned
ix = ones(1, 210); % all the observations are in the same cluster initially
centers = zeros(210, 2) + repmat(mean(data), 210, 1);
u_vec = 4 * ones(1, maxIter+1); % auxiliary variable
u = u_vec(1);
K = 1; % number of cluster

%% NGGP sampler

for iter = 1:maxIter
    for i = 1:210
        
        ptr = ix;
        ptr(i) = [];
        freqs = histcounts(ptr);
        
        % compute the probs of assigning theta_i
        prob = zeros(length(freqs) + 1, 1);
        for j = 1:length(freqs)
            if freqs(j) > 0
                prob(j) = log(freqs(j) - sigma) - ...
                    lambda_1^2/2 * (data(i,:) - centers(j,:))...
                    * (data(i,:) - centers(j,:))';
            end
        end
        prob(length(freqs)+1) = log(alpha * (u+tau)^sigma) + log(lambda_0^2) ...
            - log(lambda_0^2 + lambda_1^2) - lambda_0^2 * lambda_1^2 ...
            * data(i,:) * data(i,:)' / 2 / (lambda_0^2 + lambda_1^2);
        
        % normalize prob
        prob = prob - max(prob);
        prob = exp(prob);
        prob = prob / sum(prob);
        
        % update ix(i)
        [~, ~, ix(i)] = histcounts(rand(1), [0; cumsum(prob)]);
    end
    
    % update centers
    K = 0;
    B = accumarray(ix', 1:length(ix), [], @(x){x});
    for j = 1:length(B)
        if ~isempty(B{j})
            if size(B{j},1) == 1
                centers(j,:) = data(B{j}, :);
            else
                centers(j,:) = mean(data(B{j}, :));
            end
            K = K + 1;
        end
    end
    
    % update u
    u = u_vec(iter);
    u = u + pace * (209 / u - alpha  * (u+tau)^(sigma-1) -...
        (209 - sigma * K) / (u + tau));
    u_vec(iter+1) = u;
    
    fprintf(['iter ', num2str(iter), ' done\n'])
end

plot(u_vec)






















