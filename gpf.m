function [x_filtered,P_filtered]=gpf(N_particles,ffun,x,P,Q,hfun,z,R,U,dt)
%名称:Gaussian Particle Filter(for Gaussian noises)
%DSS模型:
%   x(n+1)=ffun(x(n),U,dt)+w
%   z(n+1)=hfun(x(n+1),U,dt)+v
%   w~N(0,Q),v~N(0,R)
%输入:
%       -N_particles:粒子数目
%       -ffun:过程函数
%       -x:前一时刻的估计值
%       -P:前一时刻的估计协方差
%       -Q:当前的过程噪声协方差
%       -hfun:观测函数
%       -z:当前的观测量
%       -R:当前的观测噪声协方差
%       -U:控制变量,处理额外信息
%       -dt:当前时刻,处理时变
%输出:
%       -x_filtered:当前的估计值
%       -P_filtered:当前的估计协方差矩阵
%

%生成以x为均值,P为协方差的粒子集x_particles
x_particles=x+chol(P)'*randn(size(x,1),N_particles);

%生成x_particles经过ffun后的粒子集fx_particles
fx_particles=zeros(size(x_particles));
for i=1:N_particles
    fx_particles(:,i)=ffun(x_particles(:,i),U,dt);
end

%fx_particles的均值与协方差
mean_fx_particles=mean(fx_particles,2);
var_fx_particles=(fx_particles-mean_fx_particles)*(fx_particles-mean_fx_particles)'/N_particles+Q;

%重新生成fx_particles
fx_particles=mean_fx_particles+chol(var_fx_particles)'*randn(size(fx_particles));

%生成fx_particles的权值weight
weight=zeros(1,N_particles);
for i=1:N_particles
    z_pre=hfun(fx_particles(:,i),U,dt);%f_particles第i个粒子对应的z的预测值
    weight(:,i)=exp(-0.5*(z-z_pre)'*pinv(R)*(z-z_pre));%粒子权重
end

%利用weight矩阵计算x_filtered,P_filtered
weight=weight/sum(weight,2);%归一化
x_filtered=fx_particles*weight';
P_filtered=(fx_particles-x_filtered)*diag(weight)*(fx_particles-x_filtered)';

end