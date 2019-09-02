% % 2d
clc;
clear;
counter=1;
h=[0.0001,0.001,0.01,0.1,1,10,100,1000]
% h= [2,4,8,16,32,64,128]
for i=h
   accResults(counter)=EvaluateModel(3,[],[],[],[],i)
   counter=counter+1;
end
figure;
plot([1:length(h)],accResults)
xticklabels(h)
ylim([0.7,1])
title('accuracy results for change lambda')
xlabel('lambda')
ylabel('accuracy')

% 2c
clc;
clear;
counter=1;
h= [2,4,8,16,32,64,128]
for i=h
   accResults(counter)=EvaluateModel(3,i,[],[],[],10)
   counter=counter+1;
end
figure;
plot([1:length(h)],accResults)
xticklabels(h)
ylim([0,1])
title('accuracy results for different size of training set')
xlabel('size of train set')
ylabel('accuracy')