%data = build_data();
%data = build_circle(sigma, num)
data = build_sphere(0.1, 240);
%plot(data_true(1,:),data_true(2,:),'-','Linewidth',4);
hold on
beta = linspace(0,2*pi,100);

%plot(q(1),q(2),'d','MarkerSize',10,'MarkerFaceColor','b');
Rho = [0.1, 0.01, 0];
Delta = [1, 10, 100,1000];
k = 20;
d = 2;
n = size(data, 2)-1;
n = 25;
Data = data;
rho = 1;
P_ = [];
TT = [];
t = tiledlayout(2,3,'TileSpacing','Compact');
nexttile
for i = 1:size(data,2)
    if mod(i,20)==0
        [~, Data_p, ~, ~] = initial_Tau(data(:,i), d, Data, k, n);  
        plot3(Data_p(1,:),Data_p(2,:),Data_p(3,:),'*');
        hold on 
    end
end
set(gca,'FontSize',18)
box on;
title('Original','FontSize',18)


Rho = cell(1,4);
for j = 1:4
    nexttile
    axis([-1.5 1.5 -4 4])
    %rho = Rho(j);
    delta = Delta(j);
    P{j} = [];
    Rho{j} = [];
    for i = 1:size(data,2)
        q = data(:,i);
        % k determine the sigma and n determines the sample size
        [Tau_p, Data_p, h, center] = initial_Tau(q, d, Data, k, n);  
        %fit_nonlinear(Data, Tau, rho, delta, adaptive, W)
        [f,rho] = fit_nonlinear(Data_p, Tau_p, rho, delta, 1);
        Rho{j} = [Rho{j},rho];
        Tau_q = projection(data(:,i), f.A, f.B, f.c, [0;0]);
        proj_q = f.Parm*Construct_Higher_Order(Tau_q);
        P{j} = [P{j}, proj_q]; 
        T = [];
        if mod(i,10)==0
            plot3(data(1,i),data(2,i),data(3,i),'d','markersize',8);
            hold on
            plot3(proj_q(1,:),proj_q(2,:),proj_q(3,:),'*','markersize',8);
            hold on
            S = ini(-0.3,0.3,10,2);
            data_new = f.Parm*Construct_Higher_Order(S);
            %plot(data_new(1,:),data_new(2,:),'-','linewidth',2);
            A = reshape_s(data_new(1,:));
            B = reshape_s(data_new(2,:));
            C = reshape_s(data_new(3,:));
            mesh(A,B,C)
            %plot3(data_new(1,:),data_new(2,:),data_new(3,:),'*');
            T = [T, Tau_q];
        end
        TT = [TT;T];
        i
        axis([-1.5 1.5 -1.5 1.5])
        set(gca,'FontSize',18)
    end
    title(['\delta=',num2str(delta)],'FontSize',18)
    box on
end


%%

nexttile
str = cell(1,5);
for i = 1:4
    A = P{i}-P{i}*diag(1./sqrt(sum(P{i}.^2,1)));
    norm_each = sum(A.^2,1);
    [C,ind] = sort(norm_each);
    if ~exist('ind','var')
        [C,ind] = sort(norm_each);
    end
    plot(norm_each(ind),'-','linewidth',2);
    hold on
    str{i} = ['\delta=',num2str(Delta(i))];
    result(i) = norm(A,'fro');
    fprintf('error%d=%.3f,rho=%.3f\n',i,result(i),mean(Rho{i}));
    set(gca,'FontSize',18)
end
str{5} = 'original';
B = data - data*diag(1./sqrt(sum(data.^2,1)));
Bs = sum(B.^2,1);
[C,ind2] = sort(Bs);
plot(C,'-','linewidth',2);
%%
legend(str)
title('fitting error','FontSize',18)
%%
% figure
% %plot(data_new(1,:),data_new(2,:),'.');
% subplot(1,2,1)
% plot(P{2}(1,:),P{2}(2,:),'o');
% subplot(1,2,2)
% plot(P{3}(1,:),P{3}(2,:),'o');
% axis([-1.5 1.5 -1.5 1.5])
%hold on
%plot(P(1,:),P(2,:),'.','MarkerSize',10)
%%
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
            inter_l = 0.001; inter_r = 1000; 
            re = search_lambda(Data, Tau, inter_l, inter_r, inter_l, delta, W);
            rho = re.rho;
        end
        
        [c, A, P] = Least_Fitting(Data, Tau, rho, W);
        d = size(Tau, 1);
        B = build_tensor(P, d);
        
        f.Data_Constructed = P*Construct_Higher_Order(Tau);
        f.A = A; f.B = B; f.c = c;

        Tau_old = Tau;
        for i = 1:size(Data, 2)
            Tau(:,i) = projection(Data(:,i), A, B, c, Tau(:,i));
        end
        Tau_ee = qrs(Tau);
        Tau = Tau_ee(1:d,:);
        f.Taus{iter} = Tau;
        f.Parm = P;
        f.Tau = Tau;
        f.Data_new_Constructed = P*Construct_Higher_Order(Tau);
        f.data_error = norm(f.Data_new_Constructed- f.Data_Constructed,'fro');
        f.Tau_error(iter) = norm(Tau'*Tau- Tau_old'*Tau_old,'fro');
        if f.Tau_error(iter) < 1.e-4 || iter>800
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


function tau = projection(x, A, B, c, tau) %project x onto f(tau) = A tau+ B(tau,tau)+c
    iter = 0; 
    while true
        Bm = tensor_fold(B, tau);
        tau_new = (2*Bm'*Bm+Bm'*A+A'*A+A'*Bm)\((2*Bm'+A')*(x-c)-Bm'*A*tau);
        if norm(tau_new-tau)<1.e-6 || iter>300
            if iter>300
                fprintf('diverge projecting tau\n');
            end
            break;
        end
        tau = tau_new;
        iter = iter+1;
    end 
end


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

function data = build_sphere(sigma, num)

    data = randn(3,num);
    %data = data*diag(1./sqrt(sum(data.^2.1)));
    data = bsxfun(@rdivide,data, sqrt(sum(data.^2,1)));
    data = data + sigma*randn(size(data));
end

