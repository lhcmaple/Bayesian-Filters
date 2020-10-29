function [xEst,PEst,xPred,PPred,zPred,inovation,S,K]=ekf(xEst,PEst,Q,ffun,fJacobian,z,R,hfun,hJacobian,U,dt)
%名称:Extended Kalman Filter
%DSS模型:
%   x(n+1)=ffun(x(n),U,dt)+w
%   z(n+1)=hfun(x(n+1),U,dt)+v
%   w~N(0,Q),v~N(0,R)
%输入:
%       -xEst:前一时刻的估计值
%       -PEst:前一时刻的估计协方差
%       -Q:当前的过程噪声协方差
%       -ffun:过程函数
%       -fJacobian:过程函数的雅可比矩阵
%       -z:当前的观测量
%       -R:当前的观测噪声协方差
%       -hfun:观测函数
%       -hJacobian:观测函数的雅可比矩阵
%       -U:控制变量,处理额外信息
%       -dt:当前时刻,处理时变
%输出:
%       -xEst:当前的估计值
%       -PEst:当前的估计协方差
%

xPred=ffun(xEst,U,dt);
F=fJacobian(xEst,U,dt);
PPred=F*PEst*F'+Q;
zPred=hfun(xPred,U,dt);
inovation=z-zPred;
H=hJacobian(xPred,U,dt);
S=H*PPred*H'+R;
K=PPred*H'*pinv(S);
xEst=xPred+K*inovation;
PEst=PPred-PPred*H'*K';

end