function [sample1,sample2] = feature_extract(signal,Fs,t1,t2)
% signal为一行信号数列，非矩阵
% Fs是采样频率(200Hz)
% t1表示选取数据片段时间长度(1s)
% t2表示数据片段选取间隔(1s)

%% 信号取段
% 正常，每行为一个样本
segment = signal_divide(signal,Fs,t1,t2);

%% 小波包分解
% 6级分解
n = 4;
N = 2^n;
wp_name = 'db4';

for i = 1:length(segment(:,1))
    wpt = wpdec(segment(i,:),n,wp_name);
    
    for j = 1:N
    % 特征1
        sample1(i,j) = norm(wpcoef(wpt,[n,j-1]),2);
    % 特征2
        sample2(i,j) = sample1(i,j).^2;
    end
    % 正常的行为样本，列为特征(235*32)
    
end

end
