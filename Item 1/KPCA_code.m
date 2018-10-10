%%%%��������س���
%�Ծ��������ɷַ�������PCA�����н�ά���õ��׻�����ͽ�ά��ľ���
%���������
%x-----������ľ���
%���������
%C-----�׻�����
%y-----��ά��ľ���

[m,n] = size(mat1);
ref_mean = mean(mat1,2);                    %�����mat1ÿ�еľ�ֵ
H =  repmat(ref_mean,1,n);
M = double(mat1);
X = M - repmat(ref_mean,1,n);            %mat1��ÿ��Ԫ�ؼ�ȥ��Ӧ�еľ�ֵ���Ա���Э����
A = X'*X;
[V D] = eig(A);                             %������ֵ����������
D_diag = diag(D);
[numd] = find(D_diag<0);                    %ȥ�����Ŷ���ɵĸ�����ֵ�����Ӧ���������� 
D_diag(numd) = [];
V(:,numd) = [];
[D_sort D_index] = sort(-D_diag);            %����ֵ����
D_sort = -D_sort;

%���ܷ����70%Ϊ��ֵ��ѡȡ��������������ֵ
D_pro = cumsum(D_sort)./sum(D_sort);
num = find(D_pro>0.9);                       %�ҵ��ۼƷ�������ܷ���70%��ֵ���ڵ�λ��
T_num = num(1);                              %��ֵ���ڵ�λ�ã�������������ֵΪ���գ�
D_select = D_sort(1:T_num);                  %ȥ����С������ֵ���õ���Ч����ֵ��������
D_index = D_index(1:T_num);                  %�õ���Ч����ֵ�����
V_select = zeros(size(V,1),T_num);
for i = 1:T_num
    V_select(:,i) = V(:,D_index(i));         %�õ���Ч��������
end
C = X*V_select*diag(1./sqrt(D_select));      %��׻�����,X*V_select��X*X'����������������diag(1./sqrt(D_select))��Ϊ�˹�һ��

%��ͶӰ���󣬼���PCA��ά��ľ���
y = C'* X;







% %����SVM֧�����������쳣�¼����з���
% clc 
% clear
% global EPS
% global C;                   % CΪ�ͷ�ֵ Ĭ��Ϊ1
% global gamma;               % 0ΪĬ��ֵȡά���ĵ���
% global L;                   % LΪ�����ĸ���
% global dim;                 % ������ά��
% global alpha;               % alpha
% global SV;                  % ֧����������һ��Ϊlabel
% global Npos;                % ���������ĸ���
% global rho;                 % �б�ʽ�е� b ֵ
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
% % ��ʾ������Ϣ
% 
% TrainFileName ='F:\����\gmcm2017\D\Q6 Code\data\y.txt';
% TestFileName  ='F:\����\gmcm2017\D\Q6 Code\data\y.txt';
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % TrainMatrixΪѵ�������ľ���
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
% % KMΪ���֮�˾���
% [KM] = KernelMatrix(TrainMatrix);
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % alphaΪһ��������
% % objΪ�Ż����Ŀ��ֵ
% % rhoΪ�б�����bֵ
% % iterΪ��������
% % Svm_Solve(TrainMatrix,KM); ʹ�õ���KM
% % Svm_Solve(TrainMatrix);    ʹ�õ���TrainMatrix
% [alpha,obj,rho,iter] = Svm_Solve(TrainMatrix,KM);
% [nSV Col] = size(find(alpha>0));
% fprintf(1,'Ŀ��ֵ = %f nSV = %d �������� = %d\n',obj,nSV,iter);
% fprintf(1,'bֵΪ  = %f \n',rho);
% toc
% tic
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Ԥ�ⲿ��
% % SVΪ֧����������Ϊ֧������֮����
% % SV_YΪ֧���������ڵ�labelֵ;
% SV = TrainMatrix(alpha>0,:);
% supportvector=find(alpha>0);
% boundvector = find(alpha==C);
% 
% alpha =alpha(alpha>0);
% TestMatrix = TrainMatrix;
% % clear TrainMatrix  % ����ѵ������
% % [TestMatrix]  = SvmReadData(TestFileName);
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % accΪ�����б𾫶�
% % outΪ����sign�б���
% % SVΪ֧������
% % TestMatrixΪ������������
% [acc_P,acc_N,Accuracy] = Svm_Predict(TestMatrix);
% fprintf(1,'\n');
% fprintf(1,'������ྫ�� = %g ������ྫ�� = %g\n',acc_P,acc_N);
% fprintf(1,'�ܷ��ྫ��   = %g       GֵΪ = %g\n',Accuracy,acc_P * acc_N);
% 
% toc
