function [xEst,x_particles,P_ekf]=epf(x_particles,P_ekf,Q,ffun,fJacobian,z,R,hfun,hJacobian,U,dt,resamplingScheme)
%名称:Extended Particle Filter(for Gaussian noises)
%DSS模型:
%   x(n+1)=ffun(x(n),U,dt)+w
%   z(n+1)=hfun(x(n+1),U,dt)+v
%   w~N(0,Q),v~N(0,R)
%输入:
%       -x_particles:前一时刻的粒子集{x_0,x_1,...,x_N_particles}
%       -P_ekf:前一时刻的粒子集对应的协方差集
%       -Q:当前的过程噪声协方差
%       -ffun:过程函数
%       -fJacobian:过程函数的雅可比矩阵
%       -z:当前的观测量
%       -R:当前的观测噪声协方差
%       -hfun:观测函数
%       -hJacobian:观测函数的雅可比矩阵
%       -U:控制变量,处理额外信息
%       -dt:当前时刻,处理时变
%       -resamplingScheme:(重采样方法)1->residualR(残差),2->systematicR(系统),default->multinomialR(多项式)
%输出:
%       -xEst:当前的估计值
%       -x_particles:当前的粒子集{x_0,x_1,...,x_N_particles}
%       -P_ekf:当前的粒子集对应的协方差集
%

if(nargin<8)
    resamplingScheme=0;
end

[nx,N_particles]=size(x_particles);
x_ekf=zeros(nx,N_particles);
x_ekf_particles=zeros(nx,N_particles);
for k=1:N_particles
    [x_ekf(:,k),P_ekf(:,:,k)]=ekf(x_particles(:,k),P_ekf(:,:,k),Q,ffun,fJacobian,z,R,hfun,hJacobian,U,dt);
    x_ekf_particles(:,k)=x_ekf(:,k)+chol(P_ekf(:,:,k))'*randn(nx,1);
end
nz=size(z,1);
z_particles=zeros(nz,N_particles);
weight=zeros(1,N_particles);
for k=1:N_particles
    z_particles(:,k)=hfun(x_ekf_particles(:,k),U,dt);
    lik=exp(-0.5*(z-z_particles(:,k))'*pinv(R)*(z-z_particles(:,k)));
    x_pred=ffun(x_particles(:,k),U,dt);
    prior=exp(-0.5*(x_ekf_particles(:,k)-x_pred)'*pinv(Q)*(x_ekf_particles(:,k)-x_pred));
    proposal=1/sqrt(det(P_ekf(:,:,k)))*exp(-0.5*(x_ekf_particles(:,k)-x_ekf(:,k))'*pinv(P_ekf(:,:,k))*(x_ekf_particles(:,k)-x_ekf(:,k)));
    weight(k)=lik*prior/proposal;
end
weight=weight/sum(weight);

if resamplingScheme==1
    outIndex=residualR(1:N_particles,weight');
elseif resamplingScheme==2
    outIndex=systematicR(1:N_particles,weight');
else
    outIndex=multinomialR(1:N,weight');
end

x_particles=x_ekf_particles(:,outIndex);
P_ekf=P_ekf(:,:,outIndex);
xEst=mean(x_particles,2);

end