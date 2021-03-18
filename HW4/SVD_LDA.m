function [model]=SVD_LDA(images,labels,input_num)
[~,num]=size(images);
%% select 2 digits
ii = 1;
jj = 1;
for i=1:num
    if labels(i)==input_num(1)
        select_num1_images(:,ii) = images(:,i);
        ii = ii+1;
    end
    if labels(i)==input_num(2)
        select_num2_images(:,jj) = images(:,i);
        jj = jj+1;
    end
end
%% SVD
select_num_images = [select_num1_images select_num2_images];
[U,S,V] = svd(select_num_images,'econ');
%% 
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
%%
sortnum1 = sort(vnum1);
sortnum2 = sort(vnum2);
t1 = length(sortnum1);
t2 = 1;
while sortnum1(t1) > sortnum2(t2)
    t1 = t1 - 1;
    t2 = t2 + 1;
end
threshold = (sortnum1(t1) + sortnum2(t2))/2;

model.U = U;
model.w = w;
model.threshold=threshold;
model.number = input_num;