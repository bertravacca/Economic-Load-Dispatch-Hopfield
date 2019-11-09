close all; clear; clc;
n=20;
rho=1;
step_size_hopfield=10^-2;
step_size_dual=10^-1;
num_dual_iter=10^2;
num_iter_hopfield=10^4;
cost_hop=NaN*zeros(1,1);
binary_cv_hop=NaN*zeros(1,1);
demand_cv_hop=NaN*zeros(1,1);
distance_hnn=NaN*zeros(1,num_dual_iter);


% We draw at random the parameters of the model
alpha=rand();
beta=rand();
p=rand(n,1);
x_0=round(rand(n,1));
y_0=rand(n,1).*x_0;
d=(1+0.1*(rand()-0.5))*sum(y_0);

%% Dual Hopfield
% Initialization
lambda_1=0;
lambda_2=0;

% define activation
temp=[100*ones(n,1);ones(n,1)];
activation=@(z) 0.5*tanh(2*temp.*(z-0.5))+0.5;
inv_activation=@(y)0.5*(1./temp).*atanh(2*y-1)+0.5;

% define objective and gradient
cost=@(x,y) p'*y+0.5*alpha*norm(x-x_0)^2+0.5*beta*norm(y-y_0)^2;
objective=@(x,y,lambda_1,lambda_2) cost(x,y)+lambda_1*(x'*y-d)+0.5*rho*(x'*y-d)^2+lambda_2*(ones(n,1)'*y-d)+0.5*rho*(ones(n,1)'*y-d)^2;
gradient=@(x,y,lambda_1,lambda_2) [alpha*(x-x_0)+lambda_1*y+rho*(y*y'*x-d*y);
    p+beta*(y-y_0)+lambda_1*x+rho*(x*x'*y-d*x)+lambda_2*ones(n,1)+rho*(ones(n,n)*y-d*ones(n,1))];
objective_hist=NaN*zeros(num_dual_iter,1);
distance_hist_geom= NaN*zeros(num_dual_iter,num_iter_hopfield);
distance_hist_L2= NaN*zeros(num_dual_iter,num_iter_hopfield);

for dual_iter=1:num_dual_iter
    z_trajectory=NaN*zeros(num_iter_hopfield,2*n);
    % apply the hopfield method
    z=0.5*ones(2*n,1);
    z_prev=z+1;
    z_h=inv_activation(z);
    for iter=1:num_iter_hopfield
        z_prev=z;
        z_h=z_h-step_size_hopfield*gradient(z(1:n),z(n+1:2*n),lambda_1,lambda_2);
        z=activation(z_h);
        z_trajectory(iter,:)=z;
    end
    objective_hist(dual_iter)=objective(z(1:n),z(n+1:2*n),lambda_1,lambda_2);
    lambda_1=lambda_1+step_size_dual*(z(1:n)'*z(n+1:2*n)-d);
    lambda_2=lambda_2+step_size_dual*(ones(n,1)'*z(n+1:2*n)-d);
    for iter=1:num_iter_hopfield
        distance_hist_geom(dual_iter,iter)=norm((asin(sqrt(z_trajectory(iter,:)'))-asin(sqrt(z)))./temp);
        distance_hist_L2(dual_iter,iter)=norm(z_trajectory(iter,:)'-z);
    end
end


figure(1)
for dual_iter=1:num_dual_iter
    semilogx(distance_hist_geom(dual_iter,:),'b')

    hold on
end

figure(2)
for dual_iter=1:num_dual_iter
    semilogx(distance_hist_L2(dual_iter,:),'r')

    hold on
end

