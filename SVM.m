
%% clear
clear all
clc
%% 导入数据
load heart_scale.mat
 
%% 随机产生训练集和测试集
n = randperm(size(heart_scale_inst,1));
 
%% 训练集――240个样本
train_matrix = heart_scale_inst(n(1:240),:);
train_label = heart_scale_label(n(1:240),:);
 
%% 测试集――30个样本
test_matrix = heart_scale_inst(n(241:end),:);
test_label = heart_scale_label(n(241:end),:);
 
%% 数据归一化
[Train_matrix,PS] = mapminmax(train_matrix');
Train_matrix = Train_matrix';
Test_matrix = mapminmax('apply',test_matrix',PS);
Test_matrix = Test_matrix';
%% SVM创建/训练(RBF核函数)
%% 寻找最佳c/g参数――交叉验证方法
[c,g] = meshgrid(-10:0.2:10,-10:0.2:10);
[m,n] = size(c);
cg = zeros(m,n);
eps = 10^(-4);
v = 5;
bestc = 1;
bestg = 0.1;
bestacc = 0;
for i = 1:m
    for j = 1:n
        cmd = ['-v ',num2str(v),' -t 2',' -c ',num2str(2^c(i,j)),' -g ',num2str(2^g(i,j))];
        cg(i,j) = svmtrain(train_label,Train_matrix,cmd);     
        if cg(i,j) > bestacc
            bestacc = cg(i,j);
            bestc = 2^c(i,j);
            bestg = 2^g(i,j);
        end        
        if abs( cg(i,j)-bestacc )<=eps && bestc > 2^c(i,j) 
            bestacc = cg(i,j);
            bestc = 2^c(i,j);
            bestg = 2^g(i,j);
        end               
    end
end
cmd = [' -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg)];
 
%% 创建/训练SVM模型
model = svmtrain(train_label,Train_matrix,cmd);
 
%% SVM仿真测试
[predict_label_1,accuracy_1,decision_values1] = svmpredict(train_label,Train_matrix,model);
[predict_label_2,accuracy_2,decision_values2] = svmpredict(test_label,Test_matrix,model)

result_1 = [train_label predict_label_1];
result_2 = [test_label predict_label_2];
%%  绘图
figure
plot(1:length(test_label),test_label,'r-*')
hold on
plot(1:length(test_label),predict_label_2,'b:o')
grid on
legend('真实类别','预测类别')
xlabel('测试集样本编号')
ylabel('测试集样本类别')
string = {'测试集SVM预测结果对比(RBF核函数)';
          ['accuracy = ' num2str(accuracy_2(1)) '%']};
title(string)

