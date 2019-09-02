function [acc] = EvaluateModel(groups, trainSize, x, y, kw, Lambda)
%% please pay attention that we chance the lambda at case 4 to lambda=lambda, for checking the check 2b
%% you can chance the the lambda to "lambda=10" or run this function with EvaluateModel(3,[],[],[],[],10)

%% set input args
if nargin <=6
    fprintf('[1] TYPICAL\n');
    fprintf('[2] SADDLE\n');
    fprintf('[3] RANDOM\n');
    fprintf('[4] RANDOM, IN ELLIPSE W/ 1 OUTLIER\n');
    fprintf('[5] SPIRAL\n');
    fprintf('[6] IMBALANCED + OVERLAP\n')
    ch = input('Choose dataset: ');             % Let the user choose
    if nargin == 1
        trainSize = [];
    end
    if nargin == 0
        ch2 = input('Choose k: ');
        groups = ch2; trainSize = [];
    end
    switch ch
        case 1
            % Set 1: TYPICAL
            kw = 0.8;   % Recommended RBF kernel width
            Lambda = Inf;    % Recommended box constraint
            x = [4,5,2,2,4,9,7,8,8,9;
                7,8,2,5,5,2,1,1,5,4]';
            y = [1 1 1 1 1 -1 -1 -1 -1 -1]';
            
        case 2
            % Set 2: SADDLE
            kw = 1;     % Recommended RBF kernel width
            Lambda = Inf;    % Recommended box constraint
            x = [4,4,2,3,8,9,7,7,5,4,6,5,8,9,6,7;
                4,6,6,3,2,3,2,0,1,0,1,2,7,8,7,5]';
            y = [1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1]';
            
        case 3
            % Set 3: RANDOM
            kw = 0.1;   % Recommended RBF kernel width
            Lambda = 1;      % Recommended box constraint
            x = 10*rand(50,2);
            y = ones(50,1); y(1:25) = -1;
            
        case 4
            % Set 4: RANDOM, IN ELLIPSE W/ 1 OUTLIER
            kw = 0.25;  % Recommended RBF kernel width
            Lambda = Lambda;     % Recommended box constraint
            x = 10*rand(150,2);
            y = (x(:,1) - 6).^2 + 3*(x(:,2) - 5).^2 - 8;
            y(y > 0) = 1; y(y ~= 1) = -1;
%             outlr = randi(150);
%             y(outlr) = -y(outlr); % Outlier (this is removable)
            
        case 5
            % Set 5: SPIRAL
            kw = 0.2;   % Recommended RBF kernel width
            Lambda = Inf;    % Recommended box constraint
            x = importdata('myspiral.mat');
            y = x(:,3); x = x(:,1:2);
            
        case 6
            % Set 6: IMBALANCED + OVERLAP
            kw = 0.5;   % Recommended RBF kernel width
            Cpos = 1;   % Recommended box constraint (red)
            Cneg = Inf; % Recommended box constraint (blue)
            x = importdata('imba.mat');
            y = x(:,3); x = x(:,1:2);
            Lambda = zeros(size(y));
            Lambda(y == 1) = Cpos; Lambda(y == -1) = Cneg;
            % Remark: Try switching Cpos and Cneg.
    end
end

% K-FOLD
[test,train] = kFold(x, y, groups, trainSize);

%% train the model
results = zeros(size(test,1),1);

for i = 1:size(test,1)
    signEval = zeros(length(test{i,1}),1);
    F = SVMtrial(train{i,1}, train{i,2}, kw, Lambda); % train
    xT = F.xT; 
    y = F.y; 
    a = F.a; 
    b = F.b; 
    kw = F.kw; 
    sv = F.sv;
    
    %evaluate the test set (validation set)
    xm = mean(test{i,1}); 
    xs = std(test{i,1});
    testx = test{i,1};
    N = length(test{i,2});
    testx = (testx - xm(ones(N,1),:))./xs(ones(N,1),:); % normalize the data
    for j = 1:length(test{i,1})
      signEval(j) = func(testx(j,:),xT,y,a,b,kw,sv); % test (function down the script)
    end
    signEval = sign(signEval);
    isRight = (signEval==test{i,2}); % evaluate
    results(i) = sum(isRight)/length(isRight); % accuracy at %
end

acc = mean(results); 

%% FUNCTION TO EVALUATE ANY UNSEEN DATA, x
%  [xT,y,a,b,kw,sv] are fixed after solving the QP.
%  f(x) = SUM_{i=sv}(y(i)*a(i)*K(x,xT(i))) + b;
    function F = func(x,xT,y,a,b,kw,sv)
        K = repmat(x,size(sv)) - xT(sv,:);      % d = (x - x')
        K = exp(-sum(K.^2,2)/kw);               % RBF: exp(-d^2/kw)
        F = sum(y(sv).*a(sv).*K) + b;           % f(x)
    end
end