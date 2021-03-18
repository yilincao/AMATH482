close all;
clear;
clc;
%% load images labels
[images, labels] = mnist_parse('train-images-idx3-ubyte', 'train-labels-idx1-ubyte');
% [test_images, test_labels] = mnist_parse('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte');
[row,col,num]=size(images);
%% reshape
reshape_images = zeros(row*col,num);
for i=1:num
    reshape_images(:,i) = reshape(images(:,:,i),row*col,1);
end
%%
[test_images, test_labels] = mnist_parse('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte');
[trow,tcol,tnum]=size(test_images);
%% reshape
treshape_images = zeros(trow*tcol,tnum);
for i=1:tnum
    treshape_images(:,i) = reshape(test_images(:,:,i),trow*tcol,1);
end
%%
ctree=fitctree(reshape_images',labels);
% view(ctree);
results = predict(ctree,treshape_images');
err = 0;
TestNum = numel(test_labels);
for i=1:TestNum
    if results(i)==test_labels(i)
        ;
    else
        err = err +1;
    end
end
sucRate = 1 - err/TestNum;
disp(['Accuracy is :',num2str(sucRate)]);