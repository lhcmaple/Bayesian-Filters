function [xEst,x_particles,P_ukf]=upf(x_particles,P_ukf,Q,ffun,z,R,hfun,U,dt,resamplingScheme,alpha,beta,kappa)
%名称:Unscented Particle Filter(for Gaussian noises)
%DSS模型:
%   x(n+1)=ffun(x(n),U,dt)+w
%   z(n+1)=hfun(x(n+1),U,dt)+v
%   w~N(0,Q),v~N(0,R)
%输入:
%       -x_particles:前一时刻的粒子集{x_0,x_1,...,x_N_particles}
%       -P_ukf:前一时刻的粒子集对应的协方差集
%       -Q:当前的过程噪声协方差
%       -ffun:过程函数
%       -z:当前的观测量
%       -R:当前的观测噪声协方差
%       -hfun:观测函数
%       -U:控制变量,处理额外信息
%       -dt:当前时刻,处理时变
%       -resamplingScheme:(重采样方法)1->residualR(残差),2->systematicR(系统),default->multinomialR(多项式)
%       -alpha,beta,kappa:参考ukf
%输出:
%       -xEst:当前的估计值
%       -x_particles:当前的粒子集{x_0,x_1,...,x_N_particles}
%       -P_ukf:当前的粒子集对应的协方差集
%

if(nargin<8)
    resamplingScheme=0;
end
if(nargin<9)
    alpha=1;
end
if(nargin<10)
    beta=0;
end
if(nargin<11)
    kappa=0;
end

[nx,N_particles]=size(x_particles);
x_ukf=zeros(nx,N_particles);
x_ukf_particles=zeros(nx,N_particles);
for k=1:N_particles
    [x_ukf(:,k),P_ukf(:,:,k)]=ukf(x_particles(:,k),P_ukf(:,:,k),U,Q,ffun,z,R,hfun,dt,alpha,beta,kappa);
    x_ukf_particles(:,k)=x_ukf(:,k)+chol(P_ukf(:,:,k))'*randn(nx,1);
end
nz=size(z,1);
z_particles=zeros(nz,N_particles);
weight=zeros(1,N_particles);
for k=1:N_particles
    z_particles(:,k)=hfun(x_ukf_particles(:,k),U,dt);
    lik=exp(-0.5*(z-z_particles(:,k))'*pinv(R)*(z-z_particles(:,k)));
    x_pred=ffun(x_particles(:,k),U,dt);
    prior=exp(-0.5*(x_ukf_particles(:,k)-x_pred)'*pinv(Q)*(x_ukf_particles(:,k)-x_pred));
    proposal=1/sqrt(det(P_ukf(:,:,k)))*exp(-0.5*(x_ukf_particles(:,k)-x_ukf(:,k))'*pinv(P_ukf(:,:,k))*(x_ukf_particles(:,k)-x_ukf(:,k)));
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

x_particles=x_ukf_particles(:,outIndex);
P_ukf=P_ukf(:,:,outIndex);
xEst=mean(x_particles,2);

end