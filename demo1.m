
%% add path
addpath('./tools/')

%% generate data
data = build_circle(0.1, 240);

%% parameter setting
p = [0.82;1.35];  k = 20;  d = 1;  n = 45;  rho = 0.01;

%%plot the original data
plot(p(1,:),p(2,:),'>','markersize',8,'MarkerFaceColor','r');
hold on

%%initalize RQMF
[Tau_p, Data_p, ~, ~] = initial_Tau(p, d, data, k, n);  

%%fitting with RQMF
[f,~] = RQMF(Data_p,Tau_p,rho,0);

%% project p onto the curve
Tau_q = projection(p, f.A, f.B, f.c, 0);
proj_q = f.Parm*Construct_Higher_Order(Tau_q);
plot(proj_q(1,:),proj_q(2,:),'ko','markersize',8,'MarkerFaceColor','k');
hold on
%% plot the curve
data_new = f.Parm*Construct_Higher_Order(linspace(-0.25,0.25,10));
plot(data_new(1,:),data_new(2,:),'b-','linewidth',2);
hold on

plot(data(1,:),data(2,:),'ro','markersize',8);
hold on
plot(Data_p(1,:),Data_p(2,:),'bd','markersize',8);