close all;
clc;
clear;
%%
% load images labels
[images, labels] = mnist_parse('train-images-idx3-ubyte', 'train-labels-idx1-ubyte');
% [test_images, test_labels] = mnist_parse('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte');
[row,col,num]=size(images);
%%
% select a rand graph to show
figure;
rand_image_inx = round(rand*num);
rand_image = images(:,:,rand_image_inx);
imshow(rand_image)
%%
% reshape
reshape_images = zeros(row*col,num);
for i=1:num
    reshape_images(:,i) = reshape(images(:,:,i),row*col,1);
end
%%
% SVD
[U,S,V] = svd(reshape_images,'econ');
figure;
for k =1:10
    subplot(2,5,k);
    ut1 = reshape(U(:,k),28,28);
    ut2 = rescale(ut1);
    imshow(ut2);
end
%% 
% singular value spectrum
figure;
subplot(2,1,1);
plot(diag(S),'ko','Linewidth',0.5,'MarkerSize',3);
set(gca,'Fontsize',12,'Xlim',[0 800]);
subplot(2,1,2);
semilogy(diag(S),'ko','Linewidth',0.5,'MarkerSize',3);
set(gca,'Fontsize',12,'Xlim',[0 800]);
%%
% plot 3D
figure;
XX = V(:,2);
YY = V(:,3);
ZZ = V(:,5);
clabels = labels+1;
scatter3(XX,YY,ZZ,clabels,clabels)
xlabel('2');
ylabel('3');
zlabel('5');
%%
