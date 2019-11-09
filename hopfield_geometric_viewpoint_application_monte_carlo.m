clear all;
num_simu=100;
n=50;
rho=1;
step_size_hopfield=10^-2;
step_size_dual=10^-1;
num_dual_iter=500;
cost_hop=NaN*zeros(num_simu,1);
binary_cv_hop=NaN*zeros(num_simu,1);
demand_cv_hop=NaN*zeros(num_simu,1);
cost_cvx=NaN*zeros(num_simu,1);
binary_cv_cvx=NaN*zeros(num_simu,1);
demand_cv_cvx=NaN*zeros(num_simu,1);

for simu=1:num_simu
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
    
    for dual_iter=1:num_dual_iter
        % apply the hopfield method
        z=0.5*ones(2*n,1);
        z_prev=z+1;
        z_h=inv_activation(z);
        for iter=1:num_dual_iter
            z_prev=z;
            z_h=z_h-step_size_hopfield*gradient(z(1:n),z(n+1:2*n),lambda_1,lambda_2);
            z=activation(z_h);
        end
        objective_hist(dual_iter)=objective(z(1:n),z(n+1:2*n),lambda_1,lambda_2);
        lambda_1=lambda_1+step_size_dual*(z(1:n)'*z(n+1:2*n)-d);
        lambda_2=lambda_2+step_size_dual*(ones(n,1)'*z(n+1:2*n)-d);
    end
    x_hop=round(z(1:n),6);
    y_hop=round(z(n+1:2*n),6);
    figure(1)
    plot(objective_hist,'r')
    title('Hopfield Surrogate Lagrangian Dual Ascent')
    
    cost_hop(simu)=cost(x_hop,y_hop);
    binary_cv_hop(simu)=(1/n)*sum((x_hop.*(1-x_hop)));
    demand_cv_hop(simu)=abs(d-round(x_hop)'*y_hop)/d;
    
    %% Convex Relaxation
    cvx_begin quiet
    variable y(n)
    minimize(p'*y+0.5*beta*square_pos(norm(y-y_0)))
    ones(1,n)*y==d
    0<=y<=1
    cvx_end
    y_cvx=round(y,6);
    x_cvx=(y_cvx~=0);
    cost_cvx(simu)=cost(x_cvx,y_cvx);
    binary_cv_cvx(simu)=(1/n)*sum(x_cvx.*(1-x_cvx));
    demand_cv_cvx(simu)=abs(d-round(x_cvx)'*y_cvx)/d;
    
    disp(['--------simulation #',num2str(simu)])
end