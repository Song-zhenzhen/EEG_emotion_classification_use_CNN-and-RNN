load('A1.mat')
for i=1:15
    for j=1:62
        A=data(i,j,:);
        B(1,:)=A;
        [sample1,sample2] = feature_extract(B,22,1,1);
        for k=1:240
            C{i,k}(j,:)=sample1(1,:);
        end
    end
end
        
