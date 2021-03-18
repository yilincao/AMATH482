close all;
clc;
clear;
%% load images labels
[images, labels] = mnist_parse('train-images-idx3-ubyte', 'train-labels-idx1-ubyte');
% [test_images, test_labels] = mnist_parse('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte');
[row,col,num]=size(images);
%% reshape
reshape_images = zeros(row*col,num);
for i=1:num
    reshape_images(:,i) = reshape(images(:,:,i),row*col,1);
end
%% select 2 digits
% select_num1 = 0;
% select_num2 = 1;
% select_num3 = 2;

select_num = [0,1,6];
k = size(select_num,2);
model = {};
for i=1:k-1
    for j =i+1:k
        model{i,j}= SVD_LDA(reshape_images,labels,[select_num(i) select_num(j)]);
    end
end

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
kk = 1;
for i=1:tnum
    if test_labels(i)==select_num(1)
        tselect_num1_images(:,ii) = treshape_images(:,i);
        ii = ii+1;
    end
    if test_labels(i)==select_num(2)
        tselect_num2_images(:,jj) = treshape_images(:,i);
        jj = jj+1;
    end
    if test_labels(i)==select_num(3)
        tselect_num3_images(:,kk) = treshape_images(:,i);
        kk = kk+1;
    end
end
%%
tselect_num_images = [tselect_num1_images tselect_num2_images tselect_num3_images];
[~,tnum1_len] = size(tselect_num1_images);
[~,tnum2_len] = size(tselect_num2_images);
[~,tnum3_len] = size(tselect_num3_images);
tlabels_num1 = ones(tnum1_len,1)*(select_num(1));
tlabels_num2 = ones(tnum2_len,1)*(select_num(2));
tlabels_num3 = ones(tnum3_len,1)*(select_num(3));
tlabels = [tlabels_num1;tlabels_num2;tlabels_num3];
results = SVD_LDA_predict(tselect_num_images,model);
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
% %%
% k = 1;
% for j = 1:TestNum
%     if results(j) ~= tlabels(j)
%         figure;
%         S = reshape(tselect_num_images(:,j),28,28);
%         imshow(S)
%         title('Wrong classified image');
%         k = k+1;
%     end
% end
