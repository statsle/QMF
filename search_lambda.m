function result = search_lambda(Data, Tau, interval_l, interval_r, epsilon, delta, W)

    if ~exist('W','var')
        W = diag(ones(1,size(Data, 2))); 
    end

    while (interval_r-interval_l) > epsilon
           middle = (interval_r+interval_l)/2;
           [g,~,~] = Eval_Rho(Data, Tau, middle, W);
           if abs(g)>delta
               interval_l = middle;
           else
               interval_r = middle;    
           end
    end
    result = (interval_r+interval_l)/2;
    
    
    function [g, t1, t2] = Eval_Rho(Data, Tau, rho, W)
            if ~exist('W','var')
                W = diag(ones(1,size(Data, 2))); 
            end
            W = W.^2;
            T= Construct_Higher_Order(Tau);
            J = Construct_Regularization(Tau);
            R = Data*W*T'/(T*W*T'+rho*J);
            g = - trace(R*J/(T*W*T'+rho*J)*J*R'); 
            t1 = norm(R*J,'fro');
            t2 = norm(R*T-Data,'fro');
    end


    function R = Construct_Regularization(Tau)
            d = size(Tau, 1);
            R = zeros(1+d+d*(d+1)/2);
            R(d+2:end,d+2:end) = eye(d*(d+1)/2);
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
end