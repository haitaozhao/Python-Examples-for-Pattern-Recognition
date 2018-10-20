function g = my_grad_minmumloss(W,X,Y)
%% Compute the gradient of minimumloss criterion
len = length(Y);
sum = 0;

for i = 1 : len
    u(i) = exp(Y(i)*W'*X(:,i));
    sum = sum + 1/(1+u(i))*1/(1+1/u(i))*Y(i)*X(:,i);
end

g = - 1/len*sum;