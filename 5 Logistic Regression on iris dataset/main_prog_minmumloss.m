clear
clc
load iris_dataset

X = irisInputs(1:2,1:100);
X = [X;ones(1,100)];
Y = [ones(50,1);-ones(50,1)];
%W = [1,0,0,0,0]'; %pinv(X')*Y;
W = pinv(X')*Y;
for i = 1 : 5000
%    alphak = btl_search(@my_fun,@my_grad,W,X,Y,0.1,0.5);
%    alp(i) = alphak;
    
    W = W - 0.1 * my_grad_minmumloss(W,X,Y);
%    f(i) = my_fun(W,X,Y);
end

for i = 1 : 100
    y_test(i) = my_sig(W,X(:,i));
end

stem(y_test)
sign = 0;
if sign ==1
figure;
%plot(X(1,51:100),X(2,51:100),'b+')
hold on
%plot(X(1,1:50),X(2,1:50),'r<')
for i = 4:0.02:7
    for j = 2:0.02:4.5
        a = [i;j;1];
        if my_sig(W,a)>0.5
            plot(i,j,'g.','MarkerSize',5);
        else
            plot(i,j,'y.','MarkerSize',5);
        end
    end
end

plot(X(1,51:100),X(2,51:100),'b+','LineWidth',2)
hold on
plot(X(1,1:50),X(2,1:50),'r<','LineWidth',2)
end
        

