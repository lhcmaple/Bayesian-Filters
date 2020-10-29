1)状态向量,观测向量,噪声向量都是列向量的形式
2)过程函数ffun,fJacobian,观测函数hfun,hJacobian的形式为:
    function [output]=fun(x,U,dt)
        %x,U,dt->output
    end
3)ukf是官方代码
4)demo_ungm是测试代码
5)参考论文:
    ukf:Unscented Filtering and Nonlinear Estimation
    upf,epf:The Unscented Particle Filter
    pf:On Sequential Monte Carlo Sampling Methods for Bayesian Filtering
    gpf:Gaussian Particle Filtering
6)一些资源:
    http://www.cs.ubc.ca/~nando/software/upf_demos.tar.gz
    http://www.cs.ubc.ca/~nando/papers/upf.ps.gz
    https://www.cs.ubc.ca/~nando/software.html
