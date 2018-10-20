clear;
clc

load iris_dataset

X = irisInputs(1:2,1:100);
X = [X;ones(1,100)];
Y = [ones(50,1);zeros(50,1)];
%W = [1,0,0,0,0]'; %pinv(X')*Y;
%W = pinv(X')*Y;
W = [1,0,0]';
for k = 1 : 50
    N1 = randperm(100);
    for i = 1 : 100
        x1 = X(:,N1(i));
        y1 = Y(N1(i));
        W = W - 0.1 * my_Sgrad(W,x1,y1);
        f((k-1)*100+i) = my_fun(W,X,Y);
    end
end
plot(f)
figure;
for i = 1 : 100
    y_test(i) = my_sig(W,X(:,i));
end

stem(y_test)
sign = 1;
if sign == 1
figure;
%plot(X(1,51:100),X(2,51:100),'b+')
hold on
%plot(X(1,1:50),X(2,1:50),'r<')
for i = 4:0.01:7
    for j = 2:0.01:4.5
        a = [i;j;1];
        if my_sig(W,a)>0.5
            plot(i,j,'b.','MarkerSize',20);
        else
            plot(i,j,'r.','MarkerSize',20);
        end
    end
end

plot(X(1,51:100),X(2,51:100),'g+')
hold on
plot(X(1,1:50),X(2,1:50),'y<')
end
        

