
function Result = QMF_projection(data, data_, k, n, d, delta ,W)
        if ~exist('W','var')
            W = diag(ones(1,size(data, 2))); %equal weight
        end
        Result = [];
        for i = 1:size(data_,2)
            q = data_(:,i);
            [Tau_p, Data_p, ~, ~] = initial_Tau(q, d, data, k, n);  
            rho = 0;
            [f,~] = fit_nonlinear(Data_p, Tau_p, rho, delta, 0);        
            %Tau_q = projection(q, f.A, f.B, f.c, zeros(d,1));
            [Tau_q,~] = gradient_descent(q, f.A, f.B, f.c, zeros(d,1));
            proj_q = f.Parm*Construct_Higher_Order(Tau_q);
            Result = [Result,proj_q];
        end      
end



%%
function [Tau, Data_selection, h, center] = initial_Tau(q, d, Data, k, n)
    [~,ind] = sort(sum((Data-q).^2,1),'ascend');
    Data_selection = Data(:,ind(2:n+1));
    
    h = find_sigma(q, Data, k);
    [U, center] = principal(Data_selection, h, q, d);
    Tau = U'*(Data-center);
  
    Tau_p = Tau(:,ind(2:n+1));
    Tau = qrs(Tau_p);
end


function [f,rho] = fit_nonlinear(Data, Tau, rho, delta, adaptive, W)
    if ~exist('W','var')
        W = diag(ones(1,size(Data, 2))); %equal weight
    end
    iter = 1;
    f.q_sequence = [];
    while true
        % Parameter for regression Using Tau
        if adaptive == 1
            inter_l = 0.001; inter_r = 1; 
            rho = search_lambda(Data, Tau, inter_l, inter_r, inter_l, delta, W);
        end
        
        [c, A, P] = Least_Fitting(Data, Tau, rho, W);
        d = size(Tau, 1);
        B = build_tensor(P, d);
        
        f.Data_Constructed = P*Construct_Higher_Order(Tau);
        f.A = A; f.B = B; f.c = c;

        Tau_old = Tau;
        for i = 1:size(Data, 2)
            [Tau(:,i),~] = gradient_descent(Data(:,i), A, B, c, Tau(:,i));
            %Tau(:,i) = projection(Data(:,i), A, B, c, Tau(:,i));
        end
        Tau_ee = qrs(Tau);
        Tau = Tau_ee(1:d,:);
        f.Taus{iter} = Tau;
        f.Parm = P;
        f.Tau = Tau;
        f.Data_new_Constructed = P*Construct_Higher_Order(Tau);
        f.data_error = norm(f.Data_new_Constructed- f.Data_Constructed,'fro');
        f.Tau_error(iter) = norm(Tau'*Tau- Tau_old'*Tau_old,'fro');
        fprintf('error1=%f,error2=%f\n',f.data_error,f.Tau_error(iter));
        if f.Tau_error(iter) < 0.03 || iter > 20 %1.e-4
            break;
        end
        iter = iter+1;
    end   
end



function theta = build_theta(Data, h, q)  
    theta = diag(sqrt(exp(-sum((Data-q).^2,1)/h^2)));
    %theta = diag(ones(1,size(Data, 2)));
end


function [c, A, P] = Least_Fitting(Data, Tau, rho, W)

    T= Construct_Higher_Order(Tau);
    d = size(Tau, 1);
    Theta = W.^2;
    R = Construct_Regularization(Tau);
    %R = Construct_Regularization2(d, T*Theta*T');
    P = Data*Theta*T'/(T*Theta*T'+rho*R);
    c = P(:,1);
    A = P(:,2:d+1);
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
    %Tau = size(Tau,2)*Q(:,2:d+1)';
    Tau = Q(:,2:d+1)';
end


function [U,center] = principal(Data, h, q, d)
    Theta = (build_theta(Data, h, q)).^2;
    center = sum(Data*Theta, 2)/sum(diag(Theta));
    [V,~,~] = svd((Data-center)*Theta*(Data-center)');
    U = V(:,1:d);
end


function B = build_tensor(para, d)
    B = zeros(size(para,1),d,d);
    ind = triu(true(d));
    for i = 1:size(para,1)
        temp = zeros(d, d);
        temp(ind) = para(i,d+2:end);
        B(i,:,:) = (temp+temp')/2;
    end
end


function [tau,Tau] = gradient_descent(x, A, B, c, tau)
    Tau = [];
    iter = 0;
    s = 1;
    while true
        Bm = tensor_fold(B, tau);
        v = x- (A*tau+ Bm*tau+c);
        g = (-A-2*Bm)'*v;
        while f_value(x, A, B, c,tau-s*g) > f_value(x, A, B, c, tau) 
            s = s/2;
        end
        tau_new = tau - s * g;
        if norm(tau_new-tau)<1.e-6 || iter>3000
            break;
        end
        tau = tau_new;
        Tau = [Tau, tau];
        iter = iter+1;
    end
end

function a = f_value(x, A, B, c, tau)
    Bm = tensor_fold(B, tau);
    a = norm(x-(A*tau+ Bm*tau+c))^2;
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

% 
% function [tau,Tau] = gradient_descent(x, A, B, c, tau)
%     Tau = [];
%     iter = 0;
%     while true
%         Bm = tensor_fold(B, tau);
%         v = x- (A*tau+ Bm*tau+c);
%         g = (-A-2*Bm)'*v;
%         tau_new = tau - 0.001 * g;
%         if norm(tau_new-tau)<1.e-6 || iter>1000
%             break;
%         end
%         tau = tau_new;
%         Tau = [Tau, tau];
%         iter = iter+1;
%     end
% end


function result = tensor_fold(B, tau)
    result = zeros(size(B,1),size(B,2));
    for i = 1:size(B,1)
        result(i,:) = squeeze(B(i,:,:))*tau;
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


function R = Construct_Regularization(Tau)

    d = size(Tau, 1);
    R = zeros(1+d+d*(d+1)/2);
    R(d+2:end,d+2:end) = eye(d*(d+1)/2);
    
    %R = eye(1+d+d*(d+1)/2);
end


function R = Construct_Regularization2(d, A)
    
    [U,~, ~] = svd(A);
    R = U(:,d+1:end)*U(:,d+1:end)';
    %R = eye(1+d+d*(d+1)/2);
end


% function theta = build_theta(Data, h, q)  
%     %theta = diag(sqrt(exp(-sum((Data-q).^2,1)/h^2)));
%     theta = diag(ones(1,size(Data, 2)));
% end


function [data_true, data] = generate_data(sigma, num)
    theta = linspace(-pi/2, pi/2, num);%pi/4:0.1:3*pi/4;
    data_true = [cos(theta);sin(theta)];
    data = data_true+sigma*randn(2,length(theta));
end


function data = build_circle(sigma, num)
    theta = linspace(-pi, pi, num);
    data = [cos(theta);sin(theta)];
    data = data + sigma*randn(size(data));
end

% function data = build_sphere(sigma, num)
% 
%     data = randn(3,num);
%     %data = data*diag(1./sqrt(sum(data.^2.1)));
%     data = bsxfun(@rdivide,data, sqrt(sum(data.^2,1)));
%     data = data + sigma*randn(size(data));
% end



function data = build_sin(sigma, num)
    theta = linspace(-pi, pi, num);
    data = [cos(theta);(theta)];
    data = data + sigma*randn(size(data));
end

function data = build_data()

    [data_true, data1] = generate_data(0.03, 50);

    [data_true, data2] = generate_data(0.03, 50);

    data = [[data1(1,:)+0.6*ones(size(data1(1,:)));data1(2,:)], data2];
end


function re = reshape_s(a)
    n = sqrt(length(a));
    re = reshape(a,[n,n]);
end

function re = ini(a,b,num,d)
    l1 = linspace(a,b,num);
    l2 = linspace(a,b,num);
    [A, B] = meshgrid(l1,l2);
    re = zeros(d,num*num);
    re(1,:) = A(:);
    re(2,:) = B(:);
end