% Assignment 5 - 2a(3a)

function [tests, trains] = kFold(X, Y, groups, trainSize)
% The kFold function divides X AND Y to k groups.
% trainSize control the size of the train group
% if no trainSize is available it will substract the test sampels and take
% the pthers for training.
% in the end, you can get a 2 cell arrays with X data and Y data for the test's groups and train's groups 
numberOfSamples = length(Y);
testSize = floor(numberOfSamples/groups);

if nargin~=4
    trainSize = numberOfSamples-testSize; % to evaluate the trainSize if not available
end

randomalization= randperm(numberOfSamples)
X=X(randomalization,:)
Y=Y(randomalization,:)

% create the cells for results
tests = cell(groups,2);
trains = cell(groups,2);
% create the groups
if groups ~= 1
   for i = 1:groups
        temp = zeros(numberOfSamples,1);
        temp([(i-1)*testSize+1:i*testSize]) = 1;
        tests{i,1} = X(temp==1,:); % the test set
        trains{i,1} = X(temp==0,:); % the rest of the data
        tests{i,2} = Y(temp==1,:); % the test set
        trains{i,2} = Y(temp==0,:); % the rest of the data
        if numberOfSamples-testSize > trainSize
            randData=randperm((numberOfSamples-testSize),trainSize)
            trains{i,1} = trains{i,1}(randData,:);
            trains{i,2} = trains{i,2}(randData,:);

        end
   end
    
   %NOT REALLY NECESSARY
% else 
%     tests{i,1} = X; 
%     trains{i,1} = X; 
%     tests{i,2} = Y; 
%     trains{i,2} = Y;
% end

end



