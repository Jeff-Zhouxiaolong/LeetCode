%%%%问题四相关程序
%对矩阵用主成分分析法（PCA）进行降维，得到白化矩阵和降维后的矩阵
%输入参量：
%x-----待处理的矩阵
%输出参量：
%C-----白化矩阵
%y-----降维后的矩阵

[m,n] = size(mat1);
ref_mean = mean(mat1,2);                    %求矩阵mat1每行的均值
H =  repmat(ref_mean,1,n);
M = double(mat1);
X = M - repmat(ref_mean,1,n);            %mat1的每个元素减去对应行的均值，以便求协方差
A = X'*X;
[V D] = eig(A);                             %求特征值和特征向量
D_diag = diag(D);
[numd] = find(D_diag<0);                    %去除因扰动造成的负特征值及其对应的特征向量 
D_diag(numd) = [];
V(:,numd) = [];
[D_sort D_index] = sort(-D_diag);            %特征值排序
D_sort = -D_sort;

%以总方差的70%为阈值，选取特征向量和特征值
D_pro = cumsum(D_sort)./sum(D_sort);
num = find(D_pro>0.9);                       %找到累计方差大于总方差70%的值所在的位置
T_num = num(1);                              %阈值所在的位置（以排序后的特征值为参照）
D_select = D_sort(1:T_num);                  %去除较小的特征值，得到有效特征值（已排序）
D_index = D_index(1:T_num);                  %得到有效特征值的序号
V_select = zeros(size(V,1),T_num);
for i = 1:T_num
    V_select(:,i) = V(:,D_index(i));         %得到有效特征向量
end
C = X*V_select*diag(1./sqrt(D_select));      %求白化矩阵,X*V_select是X*X'的特征向量，乘以diag(1./sqrt(D_select))是为了归一化

%求投影矩阵，即经PCA降维后的矩阵
y = C'* X;







% %利用SVM支持向量机对异常事件进行分类
% clc 
% clear
% global EPS
% global C;                   % C为惩罚值 默认为1
% global gamma;               % 0为默认值取维数的倒数
% global L;                   % L为样本的个数
% global dim;                 % 样本的维数
% global alpha;               % alpha
% global SV;                  % 支持向量，第一行为label
% global Npos;                % 正类样本的个数
% global rho;                 % 判别式中的 b 值
% 
% EPS = 0.00001;
% C = 100;
% gamma = 0;
% K = 0;
% L = 0;
% tic
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % 显示参数信息
% 
% TrainFileName ='F:\桌面\gmcm2017\D\Q6 Code\data\y.txt';
% TestFileName  ='F:\桌面\gmcm2017\D\Q6 Code\data\y.txt';
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % TrainMatrix为训练样本的矩阵
% [TrainMatrix] = SvmReadData(TrainFileName);
% if (gamma==0) gamma = 1 / dim; end
% 
% fprintf(1,'Total = %d\n',L);
% fprintf(1,'Class-ONE = %d\nClass-TWO = %d\n',Npos,L - Npos);
% 
% fprintf(1,'gamma =  %g \n',gamma);
% fprintf(1,'    C =  %d \n\n',C);
% 
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % KM为输出之核矩阵
% [KM] = KernelMatrix(TrainMatrix);
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % alpha为一个列向量
% % obj为优化后的目标值
% % rho为判别函数的b值
% % iter为迭代次数
% % Svm_Solve(TrainMatrix,KM); 使用的是KM
% % Svm_Solve(TrainMatrix);    使用的是TrainMatrix
% [alpha,obj,rho,iter] = Svm_Solve(TrainMatrix,KM);
% [nSV Col] = size(find(alpha>0));
% fprintf(1,'目标值 = %f nSV = %d 迭代次数 = %d\n',obj,nSV,iter);
% fprintf(1,'b值为  = %f \n',rho);
% toc
% tic
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % 预测部分
% % SV为支持向量行数为支持向量之个数
% % SV_Y为支持向量对于的label值;
% SV = TrainMatrix(alpha>0,:);
% supportvector=find(alpha>0);
% boundvector = find(alpha==C);
% 
% alpha =alpha(alpha>0);
% TestMatrix = TrainMatrix;
% % clear TrainMatrix  % 撤销训练矩阵
% % [TestMatrix]  = SvmReadData(TestFileName);
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % acc为最后的判别精度
% % out为最后的sign判别结果
% % SV为支持向量
% % TestMatrix为测试样本矩阵
% [acc_P,acc_N,Accuracy] = Svm_Predict(TestMatrix);
% fprintf(1,'\n');
% fprintf(1,'正类分类精度 = %g 负类分类精度 = %g\n',acc_P,acc_N);
% fprintf(1,'总分类精度   = %g       G值为 = %g\n',Accuracy,acc_P * acc_N);
% 
% toc
