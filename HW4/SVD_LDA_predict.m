function results=SVD_LDA_predict(images,model)
    feature = 20;
    [~,img_num] =size(images);
    [m,n] = size(model);
    if n==1
        for t = 1:img_num
            U = model{1,1}.U;
            w = model{1,1}.w;
            threshold = model{1,1}.threshold;
            select_num = model{1,1}.number;
            TestMat = U'*images(:,t);
            pval = w'*TestMat(1:feature,:);
            if pval<threshold
                ResVec = select_num(1);
            else
                ResVec = select_num(2);
            end
            results(t) = ResVec;
        end
    else
        for t = 1:img_num
            k = 1;
            for i=1:n-1
                for j = i+1:n
                    U = model{i,j}.U;
                    w = model{i,j}.w;
                    threshold = model{i,j}.threshold;
                    select_num = model{i,j}.number;
                    TestMat = U'*images(:,t);
                    pval = w'*TestMat(1:feature,:);
                    if pval<threshold
                        ResVec(k) = select_num(1);
                        k=k+1;
                    else
                        ResVec(k) = select_num(2);
                        k=k+1;
                    end
        %             ResVec(i) = (pval > threshold);
                end
            end
            results(t) = mode(ResVec);
        end
    end
results = results';

