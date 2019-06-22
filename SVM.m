
%% clear
clear all
clc
%% ��������
load heart_scale.mat
 
%% �������ѵ�����Ͳ��Լ�
n = randperm(size(heart_scale_inst,1));
 
%% ѵ�����D�D240������
train_matrix = heart_scale_inst(n(1:240),:);
train_label = heart_scale_label(n(1:240),:);
 
%% ���Լ��D�D30������
test_matrix = heart_scale_inst(n(241:end),:);
test_label = heart_scale_label(n(241:end),:);
 
%% ���ݹ�һ��
[Train_matrix,PS] = mapminmax(train_matrix');
Train_matrix = Train_matrix';
Test_matrix = mapminmax('apply',test_matrix',PS);
Test_matrix = Test_matrix';
%% SVM����/ѵ��(RBF�˺���)
%% Ѱ�����c/g�����D�D������֤����
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
 
%% ����/ѵ��SVMģ��
model = svmtrain(train_label,Train_matrix,cmd);
 
%% SVM�������
[predict_label_1,accuracy_1,decision_values1] = svmpredict(train_label,Train_matrix,model);
[predict_label_2,accuracy_2,decision_values2] = svmpredict(test_label,Test_matrix,model)

result_1 = [train_label predict_label_1];
result_2 = [test_label predict_label_2];
%%  ��ͼ
figure
plot(1:length(test_label),test_label,'r-*')
hold on
plot(1:length(test_label),predict_label_2,'b:o')
grid on
legend('��ʵ���','Ԥ�����')
xlabel('���Լ��������')
ylabel('���Լ��������')
string = {'���Լ�SVMԤ�����Ա�(RBF�˺���)';
          ['accuracy = ' num2str(accuracy_2(1)) '%']};
title(string)

