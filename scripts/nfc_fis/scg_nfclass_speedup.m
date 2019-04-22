function [fismat,outputs,recog_tr,recog_te,labels,performance]=scg_nfclass_speedup(input,target_tr,test,target_te,stepsize,class,clustsize);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In this program, the neuro-fuzzy classifier parameters are adapted by
% Scaled conjugate gradient method. Also if the gradients smoothly
% decrease, the gradients are estimated with LSE instead of directly
% calculation. This operation is satisfied to speed up the algorithm for
% medium and large scale problems.
%INPUTS
%input[N,s2]: training data
%target_tr[N,1]: the target values of training data
%test[m,s2]: test data
%target_te[m,1]: the target values of test data
%stepsize: The maximum iteration number
%class: Number of classes
%clustsize: Number of cluster of each class
%OUTPUTS
%center[s1,s2]: The center values of Gaussian functions 
%sigma_nf[s1,s2]: The width values of Gaussian functions 
%recog_tr: The recognition rate of training data 
%recog_te: The recognition rate of test data 
%out_tr[N,1]: The produced class labels of training data obtained from NFC 
%out_te[m,1]: The produced class labels of test data obtained from NFC
%performance: root mean square error of training data 

%       Written by Dr. Bayram Cetiþli Suleyman Demirel University Computer
%       Engineeering Isparta Turkey
close all;
performance=single(zeros(stepsize,1));
warning off;
rr=single(zeros(stepsize/25,4));
m=size(test,1);
[N,s2] = size(input);
%data=scale([input;test],0.1,1);
%input=data(1:N,:);test=data(N+1:end,:);
input=single(input);test=single(test);
%clear data;
target_tr=uint8(target_tr);target_te=uint8(target_te);
center=single(zeros(clustsize*class,s2));sigma_nf=single(zeros(clustsize*class,s2));
targ=single(zeros(N,class));w=single(zeros(clustsize*class,class));
sir=1;
for i=1:class
    [v,~]=find(target_tr==i);
    temp=input(v,:);   
    [idc,cc]=kmeans(temp,clustsize);
    center(sir:sir+clustsize-1,:)=cc;
    for j=1:clustsize
        ind=idc==j;
        sigma_nf(sir+j-1,:)=std(temp(ind,:));
        w(sir+j-1,i)=sum(ind)/size(v,1);
    end
    targ(v,i)=1;    
    sir=sir+clustsize;
end
ind=sigma_nf<=0;
sigma_nf(ind)=0.01;
clear ind

[s1,s2]=size(center);
X = zeros(2*s1*s2,1);
X(1:s1*s2,1)=reshape(center',s1*s2,1);
X(s1*s2+1:2*s1*s2,1)=reshape(sigma_nf',s1*s2,1);

% Initial performance
[gX,out]=grad_anfis_aralik(input,X,w,targ,class,s1,s2);
[~,tp]=max(out');out_tr=uint8(tp');
init=sum(target_tr==out_tr)/N*100;   
perf=sum(sum((single(targ)-out).^2))/N;
fprintf('initial recognation rate= %g initial perform= %g',init,perf);
fprintf('\n'); 
% Intial gradient and old gradient
gX_old = gX;

% Initial search direction and norm
dX = -gX;
nrmsqr_dX = dX'*dX;
norm_dX = sqrt(nrmsqr_dX);

% Initial training parameters and flag
sigma=5.0e-5;
lambda=5.0e-7;
success = 1;
lambdab = 0;
lambdak = lambda;
num_X=length(X);
X_tr=X';
G_tr=gX';
if (success == 1)
        sigmak = sigma/norm_dX;
        X_temp = X + sigmak*dX;
        [gX_temp,out]=grad_anfis_aralik(input,X_temp,w,targ,class,s1,s2);
        X_tr=[X_tr;X_temp'];
        G_tr=[G_tr;gX_temp'];
        sk = (gX_temp - gX)/sigmak;
        deltak = dX'*sk;    
end
% Scale deltak
deltak = deltak + (lambdak - lambdab)*nrmsqr_dX;
% IF deltak <= 0 then make the Hessian matrix positive definite
if (deltak <= 0)
    lambdab = 2*(lambdak - deltak/nrmsqr_dX);
    deltak = -deltak + lambdak*nrmsqr_dX;
    lambdak = lambdab;
end
% Calculate step size
muk = -dX'*gX;
alphak = muk/deltak;
% Calculate the comparison parameter  X_temp = X + alphak*dX;
X_temp = X + alphak*dX;        
[out,w]=output_anfis_aralik(input,X_temp,w,class,s1,s2,targ);
perf_temp=sum(sum((targ-out).^2))/N;    
difk = 2*deltak*(perf - perf_temp)/(muk^2);
% If difk >= 0 then a successful reduction in error can be made
if (difk >= 0)
    gX_old = gX;
    X = X_temp;             
    [gX,out]=grad_anfis_aralik(input,X_temp,w,targ,class,s1,s2);        
    X_tr=[X_tr;X'];
    G_tr=[G_tr;gX'];
    perf_temp=sum(sum((targ-out).^2))/N;
    % Initial gradient and old gradient
    lambdab = 0;
    success = 1;
    perf = perf_temp;  
    % Restart the algorithm every num_X iterations
    dX = -gX;       
    nrmsqr_dX = dX'*dX;
    norm_dX = sqrt(nrmsqr_dX);        
    % If difk >= 0.75, then reduce the scale parameter
    if (difk >= 0.75)
        lambdak = 0.25*lambdak;
    end
else
    lambdab = lambdak;
    success = 0;
end
% If difk < 0.25, then increase the scale parameter
if (difk < 0.25)
    lambdak = lambdak + deltak*(1 - difk)/nrmsqr_dX;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%The training of NFC with SCG algorithm
for epoch=1:stepsize        
    % If success is true, calculate second order information
    if (success == 1)
        sigmak = sigma/norm_dX;
        X_temp = X + sigmak*dX;         
        ind=X_temp(s1*s2+1:2*s1*s2)<=0;
        X_temp(ind)=0.01;
        [gX_temp]=lse_gradient(num_X,X_temp,X_tr,G_tr);        
        if (find(isinf(gX_temp))>=1) | (find(isnan(gX_temp))>=1)
            fprintf('epoch %g  the gradient is calculated as directly %g \n',epoch);
            fprintf('\n');
            if isempty(X_temp)                 
                [gX_temp,out]=grad_anfis_aralik(input,X_temp,w,targ,class,s1,s2); 
            else
                X_tr=[X_temp;X_tr];
                G_tr=[gX_temp;G_tr];
                [gX_temp]=lse_gradient(num_X,X_temp,X_tr,G_tr);
            end                                
        end    
        sk = (gX_temp - gX)/sigmak;
        deltak = dX'*sk;        
    end
    % Scale deltak
    deltak = deltak + (lambdak - lambdab)*nrmsqr_dX;
    % IF deltak <= 0 then make the Hessian matrix positive definite
    if (deltak <= 0)
        lambdab = 2*(lambdak - deltak/nrmsqr_dX);
        deltak = -deltak + lambdak*nrmsqr_dX;
        lambdak = lambdab;
    end
    % Calculate step size
    muk = -dX'*gX;
    alphak = muk/deltak;
    if muk==0
        break;
    end
    
    % Calculate the comparison parameter  X_temp = X + alphak*dX;
    X_temp = X + alphak*dX;    
    ind=X_temp(s1*s2+1:2*s1*s2)<=0;
    X_temp(ind)=0.01;       
    [out,w]=output_anfis_aralik(input,X_temp,w,class,s1,s2,targ);
    perf_temp=sum(sum((single(targ)-out).^2))/N;    
    difk = 2*deltak*(perf - perf_temp)/(muk^2);
    
    % If difk >= 0 then a successful reduction in error can be made
    if (difk >= 0)
        gX_old = gX;
        X = X_temp;           
        ind=X(s1*s2+1:2*s1*s2)<=0;
        X(ind)=0.01;
        [gX,out]=grad_anfis_aralik(input,X,w,targ,class,s1,s2);        
        if size(X_tr,1)==3
            x_temp=X_tr(1,:);
            X_tr(1,:)=[];
            g_temp=G_tr(1,:);
            G_tr(1,:)=[];
        end
        X_tr=[X_tr;X_temp'];
        G_tr=[G_tr;gX'];             
        perf_temp=sum(sum((single(targ)-out).^2))/N;
        % Initial gradient and old gradient
        lambdab = 0;
        success = 1;
        perf = perf_temp;
        
        % Restart the algorithm every num_X iterations
        if rem(epoch,num_X)==0
            dX = -gX;
        else
            betak = (gX'*gX - gX'*gX_old)/muk;
            dX = -gX + betak*dX;
        end
        nrmsqr_dX = dX'*dX;
        norm_dX = sqrt(nrmsqr_dX);
        
        % If difk >= 0.75, then reduce the scale parameter
        if (difk >= 0.75)
            lambdak = 0.25*lambdak;
        end
    else
        lambdab = lambdak;
        success = 0;
    end
    
    % If difk < 0.25, then increase the scale parameter
    if (difk < 0.25)
        lambdak_old=lambdak;
        lambdak = lambdak + deltak*(1 - difk)/nrmsqr_dX;
        if isinf(lambdak)
            lambdak=lambdak_old*1.2;
        end
    end
    performance(epoch,1)=perf;
    if rem(epoch,25)==0 | rem(epoch,stepsize)==0
        [rr,recog_tr,recog_te,out_tr,out_te]=performance_measurement(rr,epoch,perf_temp,out,target_tr,test,target_te,X,w,class,s1,s2);               
        file=['SCG_NFC_results_',date];
        save (file, 'rr', 'recog_tr', 'recog_te', 'X', 'w');
        if rr(end-1,4)==rr(end,4)
            fprintf('\n');
            disp('The gradient does not change, and the program is broken');
            break;
        end
    end 
    
end
center=(reshape(X(1:s1*s2,1),s2,s1))';
sigma_nf=(reshape(X(s1*s2+1:2*s1*s2,1),s2,s1))'; 
outputs.center=center;
outputs.sigma_nf=sigma_nf;
outputs.w=w;
labels.input=out_tr;
labels.test=out_te;
figure;plot(performance);
title('Performance evaluation');
xlabel('Epochs');
ylabel('RMSE value');
fismat=nfc_fis(double(input),double(target_tr),double(center),double(sigma_nf),class,clustsize);
ruleview(fismat);

%Fonctions
%**************************************************************************
%Calculation of gradients and output values together
function [gX, out]=grad_anfis_aralik(input,X,w,targ,class,s1,s2)
N=size(input,1);
mem=single(zeros(N,s1));

for i=1:s1
    mem(:,i)=exp(sum(-0.5*[(input-ones(N,1)*X((i-1)*s2+1:i*s2,1)').^2]./(ones(N,1)*X((i-1)*s2+s1*s2+1:i*s2+s1*s2,1)').^2,2));    
end
out_t=mem*w;
top=single(sum(out_t,2));
ind=top==0;
top(ind)=0.01;
out=out_t./(top*ones(1,class));
gX=single(zeros(2*s1*s2,1));
t1=-2*(single(targ)-out);
sira=1;sir=1;
for k=1:class
    for j=1:s1/class  
        temp=[mem(:,sira).*t1(:,k).*(1-out(:,k))./top(:,1)*w(sira,k)];
        gX(sir:sir+s2-1)=gX(sir:sir+s2-1)+[(input-ones(N,1)*X((sira-1)*s2+1:sira*s2,1)')./(ones(N,1)*X((sira-1)*s2+s1*s2+1:sira*s2+s1*s2,1)'.^2)]'*temp;
        gX(s1*s2+sir:s1*s2+sir+s2-1)=gX(s1*s2+sir:s1*s2+sir+s2-1)+[(input-ones(N,1)*X((sira-1)*s2+1:sira*s2,1)').^2./(ones(N,1)*X((sira-1)*s2+s1*s2+1:sira*s2+s1*s2,1)'.^3)]'*temp;        
        sir=sir+s2;sira=sira+1;
    end
end
gX=gX/N;

%**************************************************************************
%The calculation of only output values
function [out,w]=output_anfis_aralik(input,X,w,class,s1,s2,targ)
if nargin<7
    targ=[];
end
N=size(input,1);
mem=single(zeros(N,s1));
for i=1:s1
    mem(:,i)=exp(sum(-0.5*[(input-ones(N,1)*X((i-1)*s2+1:i*s2,1)').^2]./(ones(N,1)*X((i-1)*s2+s1*s2+1:i*s2+s1*s2,1)').^2,2));   
end
if isempty(targ)==0 && s1/class>1
    for i=1:class
        [v,vv]=find(targ(:,i)==1);
        [~,vv1]=max(mem(v,[(i-1)*s1/class+1:i*s1/class])');
        for j=1:s1/class
            w((i-1)*s1/class+j,i)=sum(vv1==j)/size(v,1);
        end
    end
end
out_t=mem*w;
top=single(sum(out_t,2));
ind=top==0;
top(ind)=0.01;
out=out_t./(top*ones(1,class));



%*************************************************************************
function [rr,recog_tr,recog_te,out_tr,out_te]=performance_measurement(rr,epoch,perf_temp,out,target_tr,test,target_te,X,w,class,s1,s2);
N=size(target_tr,1);
m=size(target_te,1);
[~,tp]=max(out');out_tr=uint8(tp');
indx=(out_tr==target_tr);
recog_tr=sum(indx)/N*100;
output=output_anfis_aralik(test,X,w,class,s1,s2);
[~,tp]=max(output');out_te=uint8(tp');
indx=(out_te==target_te);
recog_te=sum(indx)/m*100;
fprintf('epoch %g   recog_train  %g  recog_test  %g  performance   %g\n',epoch,recog_tr,recog_te,perf_temp);
rr=[rr;epoch recog_tr recog_te perf_temp];
if recog_te>99.5
    return;
end


%*************************************************************************
function [gX_temp]=lse_gradient(num_X,X_temp,X_tr,G_tr)
for i=1:num_X
    P=[X_tr(:,i).^2 X_tr(:,i) ones(size(X_tr,1),1)]\G_tr(:,i);
    if isnan(P(1,1))==1 || isinf(P(1,1))==1
        gX_temp(i,1)=G_tr(end,i);
    else
        gX_temp(i,1)=[X_temp(i,1)^2 X_temp(i,1) 1]*P;
    end
end
