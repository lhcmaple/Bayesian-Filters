function [xEst,x_particles]=pf(x_particles,Q,ffun,z,R,hfun,U,dt,resamplingScheme)
%名称:Particle Filter(for Gaussian noises),SIR
%DSS模型:
%   x(n+1)=ffun(x(n),U,dt)+w
%   z(n+1)=hfun(x(n+1),U,dt)+v
%   w~N(0,Q),v~N(0,R)
%输入:
%       -x_particles:前一时刻的粒子集{x_0,x_1,...,x_N_particles}
%       -Q:当前的过程噪声协方差
%       -ffun:过程函数
%       -z:当前的观测量
%       -R:当前的观测噪声协方差
%       -hfun:观测函数
%       -U:控制变量,处理额外信息
%       -dt:当前时刻,处理时变
%       -resamplingScheme:(重采样方法)1->residualR(残差),2->systematicR(系统),default->multinomialR(多项式)
%输出:
%       -xEst:当前的估计值
%       -x_particles:当前的粒子集{x_0,x_1,...,x_N_particles}
%

if(nargin<9)
    resamplingScheme=0;
end
[nx,N_particles]=size(x_particles);
for k=1:N_particles
    x_particles(:,k)=ffun(x_particles(:,k),U,dt)+chol(Q)'*randn(nx,1);
end
nz=size(z,1);
z_particles=zeros(nz,N_particles);
weight=zeros(1,N_particles);
for k=1:N_particles
    z_particles(:,k)=hfun(x_particles(:,k),U,dt);
    weight(k)=exp(-0.5*(z-z_particles(:,k))'*pinv(R)*(z-z_particles(:,k)));
end
weight=weight/sum(weight);

if resamplingScheme==1
    outIndex=residualR(1:N_particles,weight');
elseif resamplingScheme==2
    outIndex=systematicR(1:N_particles,weight');
else
    outIndex=multinomialR(1:N,weight');
end

x_particles=x_particles(:,outIndex);
xEst=mean(x_particles,2);

end