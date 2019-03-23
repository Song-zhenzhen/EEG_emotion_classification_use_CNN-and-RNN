function [segment] = signal_divide(signal,Fs,t1,t2)
% (无误)
% signal原始信号
% Fs是采样频率
% t1表示选取数据片段时间长度
% t2表示数据片段选取间隔

j = 0;
for i = 1:fix(((length(signal)/Fs-t1)/t2)+1) %(length(s)/Fs)为整个数据的时间长度，在除以t2后为整个采样的数据总数
    segment(i,:) = signal(round(j*Fs+1):round((t1+j)*Fs));%235*200，每行为一个样本
    j = t2*i;
end

end