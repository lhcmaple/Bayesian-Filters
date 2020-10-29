function [xEst,PEst,xPred,PPred,zPred,inovation,S,K]=kf(xEst,PEst,Q,F,z,R,H,U)
%名称:Kalman Filter
%DSS模型:
%   x(n+1)=F*x(n)+U+w
%   z(n+1)=H*x(n+1)+v
%   w~N(0,Q),v~N(0,R)
%输入:
%       -xEst:前一时刻的估计值
%       -PEst:前一时刻的估计协方差
%       -Q:当前的过程噪声协方差
%       -F:当前的过程矩阵
%       -z:当前的观测量
%       -R:当前的观测噪声协方差
%       -H:当前的观测矩阵
%       -U:控制变量,处理额外信息
%输出:
%       -xEst:当前的估计值
%       -PEst:当前的估计协方差
%

if nargin<8
    U=zeros(size(xEst));
end

xPred=F*xEst+U;
zPred=H*xPred;
inovation=z-zPred;
PPred=F*PEst*F'+Q;
S=H*PPred*H'+R;
K=PPred*H'*pinv(S);
xEst=xPred+K*inovation;
PEst=PPred-PPred*H'*K';

end