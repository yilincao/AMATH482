close all;
clc;
clear;
%% LDA
% load images labels
[images, labels] = mnist_parse('train-images-idx3-ubyte', 'train-labels-idx1-ubyte');
% [test_images, test_labels] = mnist_parse('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte');
[row,col,num]=size(images);
% reshape
reshape_images = zeros(row*col,num);
for i=1:num
    reshape_images(:,i) = reshape(images(:,:,i),row*col,1);
end
% select 2 digits
select_num = [0,1];
select_num1 = select_num(1);
select_num2 = select_num(2);
ii = 1;
jj = 1;
for i=1:num
    if labels(i)==select_num1
        select_num1_images(:,ii) = reshape_images(:,i);
        ii = ii+1;
    end
    if labels(i)==select_num2
        select_num2_images(:,jj) = reshape_images(:,i);
        jj = jj+1;
    end
end
%%
[~,num1_len] = size(select_num1_images);
[~,num2_len] = size(select_num2_images);
labels_num1 = ones(num1_len,1)*(select_num1);
labels_num2 = ones(num2_len,1)*(select_num2);
select_num_labels = [labels_num1;labels_num2];
select_num_images = [select_num1_images select_num2_images];
%%
select_num1 = select_num(1);
select_num2 = select_num(2);
ii = 1;
jj = 1;
for i=1:num
    if labels(i)==select_num1
        select_num1_images(:,ii) = reshape_images(:,i);
        ii = ii+1;
    end
    if labels(i)==select_num2
        select_num2_images(:,jj) = reshape_images(:,i);
        jj = jj+1;
    end
end
%%
[~,num1_len] = size(select_num1_images);
[~,num2_len] = size(select_num2_images);
labels_num1 = ones(num1_len,1)*(select_num1);
labels_num2 = ones(num2_len,1)*(select_num2);
select_num_labels = [labels_num1;labels_num2];
select_num_images = [select_num1_images select_num2_images];
%%
[test_images, test_labels] = mnist_parse('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte');
[trow,tcol,tnum]=size(test_images);
%% reshape
treshape_images = zeros(trow*tcol,tnum);
for i=1:tnum
    treshape_images(:,i) = reshape(test_images(:,:,i),trow*tcol,1);
end
%% select 2 digits
ii = 1;
jj = 1;
for i=1:tnum
    if test_labels(i)==select_num1
        tselect_num1_images(:,ii) = treshape_images(:,i);
        ii = ii+1;
    end
    if test_labels(i)==select_num2
        tselect_num2_images(:,jj) = treshape_images(:,i);
        jj = jj+1;
    end
end
%%
tselect_num_images = [tselect_num1_images tselect_num2_images];
[~,tnum1_len] = size(tselect_num1_images);
[~,tnum2_len] = size(tselect_num2_images);
tlabels_num1 = ones(tnum1_len,1)*(select_num1);
tlabels_num2 = ones(tnum2_len,1)*(select_num2);
tlabels = [tlabels_num1;tlabels_num2];
%% SVD
[U,S,V] = svd(select_num_images,'econ');
%
feature = 20;
digits = S*V';
n1 = size(select_num1_images,2);
n2 = size(select_num2_images,2);
num1 = digits(1:feature,1:n1);
num2 = digits(1:feature,n1+1:n1+n2);
m1 = mean(num1,2);
m2 = mean(num2,2);
Sw = 0; % within class variances
for k = 1:n1
    Sw = Sw + (num1(:,k) - m1)*(num1(:,k) - m1)';
end
for k = 1:n2
    Sw = Sw + (num2(:,k) - m2)*(num2(:,k) - m2)';
end
Sb = (m1-m2)*(m1-m2)'; % between class
[V2, D] = eig(Sb,Sw); % linear disciminant analysis
[lambda, ind] = max(abs(diag(D)));
w = V2(:,ind);
w = w/norm(w,2);
vnum1 = w'*num1;
vnum2 = w'*num2;
if mean(vnum1) > mean(vnum2)
    w = -w;
    vnum1 = -vnum1;
    vnum2 = -vnum2;
end
%
sortnum1 = sort(vnum1);
sortnum2 = sort(vnum2);
t1 = length(sortnum1);
t2 = 1;
while sortnum1(t1) > sortnum2(t2)
    t1 = t1 - 1;
    t2 = t2 + 1;
end
threshold = (sortnum1(t1) + sortnum2(t2))/2;
%
TestNum = size(tselect_num_images,2);
for t = 1:TestNum
    TestMat = U'*tselect_num_images(:,t);
    pval = w'*TestMat(1:feature,:);
    if pval<threshold
        ResVec = select_num(1);
    else
        ResVec = select_num(2);
    end
    results(t) = ResVec;
end

err = 0;
TestNum = numel(tlabels);
for num_i=1:TestNum
    if results(num_i)==tlabels(num_i)
        ;
    else
        err = err +1;
    end
end
sucRate = 1 - err/TestNum;
disp(['Accuracy is :',num2str(sucRate)]);
%% SVM classifier with training data, labels and test set
%
tic;
tTree = templateTree('surrogate','on');
tEnsemble = templateEnsemble('GentleBoost',100,tTree);
options = statset('UseParallel',true);
Mdl = fitcecoc(select_num_images',select_num_labels,'Coding','onevsall','Learners',tEnsemble,...
                'Prior','uniform','Options',options);%,'NumBins',50
toc;
%
testLength = length(tlabels);
testResults = -1*ones(testLength,1);
parfor i=1:testLength
    testResults(i) = predict(Mdl,tselect_num_images(:,i)');
end
% Mdl = fitcecoc(reshape_images',labels);
% test_results = predict(Mdl,treshape_images);
err = 0;
TestNum = numel(tlabels);
for i=1:TestNum
    if testResults(i)==tlabels(i)
        ;
    else
        err = err +1;
    end
end
sucRate = 1 - err/TestNum;
disp(['Accuracy is :',num2str(sucRate)]);
%
ctree=fitctree(select_num_images',select_num_labels);
% view(ctree);
results = predict(ctree,tselect_num_images');
err = 0;
TestNum = numel(tlabels);
for i=1:TestNum
    if results(i)==tlabels(i)
        ;
    else
        err = err +1;
    end
end
sucRate = 1 - err/TestNum;
disp(['Accuracy is :',num2str(sucRate)]);
