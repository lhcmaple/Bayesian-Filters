%UNGM测试
clear;clc;close all;
x0=0.1;
sigma_u=1;
sigma_v=1;
alpha=0.5;
beta=10;
gamma=8;
N=100;
h=@(x,U,dt) (x^2/20);
f=@(x,U,dt) (alpha*x+beta*x/(1+x^2)+gamma*cos(1.2*(dt-1)));
fJ=@(x,U,dt) alpha+beta*((1+x^2)-2*x^2)/(1+x^2)^2;
hJ=@(x,U,dt) x/10;
P=1;
mc_times=50;
M=100;
mean_dx=zeros(1,mc_times);
mean_dx_ukf=zeros(1,mc_times);
mean_dx_ekf=zeros(1,mc_times);
mean_dx_upf=zeros(1,mc_times);
for i=1:mc_times
    randn('seed',i);
    x=zeros(1,N);
    x(1)=x0+sqrt(P)*randn;
    for k=2:N
        x(k)=f(x(k-1),[],k)+sigma_u*randn;
    end

    z=zeros(1,N);
    for k=1:N
        z(k)=h(x(k),[],k)+sigma_v*randn;
    end

    x_filtered=zeros(1,N);
    x_filtered(1)=x0;
    P_filtered=P;
    x_particles=x0+chol(P_filtered)'*randn(size(x0,1),M);
    x_filtered_ukf=zeros(1,N);
    x_filtered_ukf(1)=x0;
    P_filtered_ukf=P;
    x_filtered_ekf=zeros(1,N);
    x_filtered_ekf(1)=x0;
    P_filtered_ekf=P;
    x_filtered_upf=zeros(1,N);
    x_particles_upf=x0+chol(P_filtered)'*randn(size(x0,1),M);
    P_filtered_upf=ones(1,1,M);
    for k=2:N
        [x_filtered(k),x_particles]=pf(x_particles,sigma_u^2,f,z(k),sigma_v^2,h,[],k,2);
        [x_filtered_ukf(k),P_filtered_ukf]=ukf(x_filtered_ukf(k-1),P_filtered_ukf,[],sigma_u^2,f,z(k),sigma_v^2,h,k,1.2,.88,0);
        [x_filtered_ekf(k),P_filtered_ekf]=ekf(x_filtered_ekf(k-1),P_filtered_ekf,sigma_u^2,f,fJ,z(k),sigma_v^2,h,hJ,[],k);
        [x_filtered_upf(k),x_particles_upf,P_filtered_upf]=upf(x_particles_upf,P_filtered_upf,sigma_u^2,f,z(k),sigma_v^2,h,[],k,2,1,0,2);
%         [x_filtered_upf(k),x_particles_upf,P_filtered_upf]=epf(x_particles_upf,P_filtered_upf,sigma_u^2,f,fJ,z(k),sigma_v^2,h,hJ,[],k,2);
    end
    disp(i);
    mean_dx(i)=sum((x_filtered-x).^2)/N;
    mean_dx_ukf(i)=sum((x_filtered_ukf-x).^2)/N;
    mean_dx_ekf(i)=sum((x_filtered_ekf-x).^2)/N;
    mean_dx_upf(i)=sum((x_filtered_upf-x).^2)/N;
end

figure(1);
hold on;
plot(1:100,x(1:100),'.-');
plot(1:100,x_filtered(1:100),'o--');
plot(1:100,x_filtered_ukf(1:100),'*--');
plot(1:100,x_filtered_ekf(1:100),'s--');
plot(1:100,x_filtered_upf(1:100),'^--');
legend('x','pf','ukf','ekf','upf');
axis([0,50,-20,20]);

figure(2);
hold on;
title('每次仿真的平均MSE');
plot(1:mc_times,mean_dx,'o-');
plot(1:mc_times,mean_dx_ukf,'*-');
plot(1:mc_times,mean_dx_ekf,'s-');
plot(1:mc_times,mean_dx_upf,'^-');
legend('pf','ukf','ekf','upf');
xlabel('仿真次数');
ylabel('RMSE');
fprintf(1,'50次仿真的平均MSE\nekf:%f\nukf:%f\npf:%f\nupf:%f\n',mean(mean_dx_ekf),mean(mean_dx_ukf),mean(mean_dx),mean(mean_dx_upf));
axis([0,50,0,3]);
saveas(2,'demo_ungm/每次仿真的平均RMSE.png');