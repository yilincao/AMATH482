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
%%
[test_images, test_labels] = mnist_parse('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte');
[trow,tcol,tnum]=size(test_images);
%% reshape
treshape_images = zeros(trow*tcol,tnum);
for i=1:tnum
    treshape_images(:,i) = reshape(test_images(:,:,i),trow*tcol,1);
end
%%
select_num = [0,1,2,3,4,5,6,7,8,9];
k = size(select_num,2);
for i=1:k-1
    for j =i+1:k
        model= SVD_LDA(reshape_images,labels,[select_num(i),select_num(j)]);
        % select 2 digits,clear selected number
        clear tselect_num1_images;
        clear tselect_num2_images;
        clear tlabels;
        ii = 1;
        jj = 1;
        for indx=1:tnum
            if test_labels(indx)==select_num(i)
                tselect_num1_images(:,ii) = treshape_images(:,indx);
                ii = ii+1;
            end
            if test_labels(indx)==select_num(j)
                tselect_num2_images(:,jj) = treshape_images(:,indx);
                jj = jj+1;
            end
        end
        %%
        tselect_num_images = [tselect_num1_images tselect_num2_images];
        [~,tnum1_len] = size(tselect_num1_images);
        [~,tnum2_len] = size(tselect_num2_images);
        tlabels_num1 = ones(tnum1_len,1)*select_num(i);
        tlabels_num2 = ones(tnum2_len,1)*select_num(j);
        tlabels = [tlabels_num1;tlabels_num2];
        %%
        results = SVD_LDA_predict(tselect_num_images,{model});
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
        sucRateall(i,j)=sucRate;
    end
end
max_acc = 0;
max_row = 1;
max_col = 1;
min_acc = 1;
min_row = 1;
min_col = 1;
for i=1:k-1
    for j =i+1:k
        if max_acc<sucRateall(i,j)
            max_acc = sucRateall(i,j);
            max_row =select_num(i);
            max_col = select_num(j);
        end
        if min_acc>sucRateall(i,j)
            min_acc = sucRateall(i,j);
            min_row =select_num(i);
            min_col = select_num(j);
        end
    end
end
disp(['Maximum Accuracy is :',num2str(max_acc)]);
disp(['Number ',num2str(max_row),' and ',num2str(max_col)]);
disp(['Minimum Accuracy is :',num2str(min_acc)]);
disp(['Number ',num2str(min_row),' and ',num2str(min_col)]);
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
