clear all;
close all;
iris = csvread('/Users/StasDon/git/musicgenrerecognition/data/GTZAN/data2_l20feat.csv',1,0);

FEATURE_N = 20;
CLASS_N = 10;

input=iris(1:3:end,1:FEATURE_N);
test=iris(3:4:end,1:FEATURE_N);
target_tr=iris(1:3:end,FEATURE_N + 1);
target_te=iris(3:4:end,FEATURE_N + 1);

[fismat4,feature,outputs,recog_tr,recog_te,labels,performance]=nfc_feature_select([input;test],[target_tr;target_te],test,target_te,1000,CLASS_N);

m_matrix = [labels.test(:), target_te(:)];
display(m_matrix)

cp = classperf(labels.test(:), target_te(:));
fprintf('Accuracy: %g',1 - cp.ErrorRate);
fprintf('\n'); 
