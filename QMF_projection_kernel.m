function Result = QMF_projection_kernel(data, data_, k,  d, delta , W2)
        if ~exist('W2','var')
            W2 = diag(ones(1,size(data, 2))); %equal weight
        end
        Result = [];
        for i = 1:size(data_,2)
            q = data_(:,i);
            [Tau, W1] = inital_Tau(q, d, data, k);
            rho = 0.1;
            W = W1*W2;
            [f,~] = fit_nonlinear(data, Tau, rho, delta, 1, W);
            Tau_q = projection(q, f.A, f.B, f.c, zeros(d,1));
            proj_q = f.Parm*Construct_Higher_Order(Tau_q);
            Result = [Result,proj_q];
        end      
end


function [Tau, W] = inital_Tau(q, d, data, k)
    h = find_sigma(q, data, k);
    W = build_theta(data, h, q);
    [U, center] = principal(data, h, q, d);
    Tau = U'*(data-center);
    Tau = qrs(Tau);
end


function theta = build_theta(Data, h, q)  
    theta = diag(sqrt(exp(-sum((Data-q).^2,1)/h^2)));
end


function sigma = find_sigma(x, Data, k)
    s_distance = sum((Data-x).^2, 1);
    [~,ind] = sort(s_distance,'ascend');
    Neig = Data(:,ind(2:k+1)); 
    sigma = max(sqrt(sum((Neig-x).^2,1)));
end


function Tau = qrs(Tau)
    d = size(Tau, 1);
    [Q,~] = qr([ones(size(Tau, 2),1),Tau']);
    Tau = Q(:,2:d+1)';
end


function [U,center] = principal(Data, h, q, d)
    Theta = (build_theta(Data, h, q)).^2;
    center = sum(Data*Theta, 2)/sum(diag(Theta));
    [V,~,~] = svd((Data-center)*(Theta.^2)*(Data-center)');
    U = V(:,1:d);
end


function tau = projection(x, A, B, c, tau) %project x onto f(tau) = A tau+ B(tau,tau)+c
    iter = 0; 
    while true
        Bm = tensor_fold(B, tau);
        tau_new = (2*Bm'*Bm+Bm'*A+A'*A+A'*Bm)\((2*Bm'+A')*(x-c)-Bm'*A*tau);
        if norm(tau_new-tau)<1.e-6 || iter>300
%             if iter>300
%                 fprintf('diverge projecting tau\n');
%             end
            break;
        end
        tau = tau_new;
        iter = iter+1;
    end 
end


function T = Construct_Higher_Order(Tau) 
    d = size(Tau, 1);
    T = zeros(1+d+d*(d+1)/2, size(Tau,2));
    ind = triu(true(size(Tau, 1)));
    for i = 1:size(Tau,2)
        T(1:1+d,i) = [1; Tau(:,i)];
        temp = Tau(:,i)*Tau(:,i)';
        T(d+2:end,i) = temp(ind);
    end
end