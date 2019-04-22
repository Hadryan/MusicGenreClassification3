function [fismat,outputs,recog_tr,recog_te,labels,performance]=scg_nfc(input,target_tr,test,target_te,stepsize,class,clustsize);
% In this program, the neuro-fuzzy classifier parameters are adapted by Scaled conjugate gradient method.

%INPUTS
%input[N,s2]: training data
%target_tr[N,1]: the target_tr values of training data
%test[m,s2]: test data
%target_te[m,1]: the target values of test data
%stepsize: The maximum iteration number
%class: Number of classes
%clustsize: Number of cluster of each class
%OUTPUTS
%outputs.center[s1,s2]: The center values of Gaussian functions 
%outputs.sigma_nf[s1,s2]: The width values of Gaussian functions 
%recog_tr: The recognition rate of training data 
%recog_te: The recognition rate of test data 
%labels.input=out_tr[N,1]: The produced class labels of training data obtained from NFC 
%labels.test=out_te[m,1]: The produced class labels of test data obtained from NFC
%performance: root mean square error of training data 
%fismat: Demonstration of NFC in fuzzy viewer.
%       Written by Dr. Bayram Ceti�li Suleyman Demirel University Computer
%       Engineeering Isparta Turkey

warning off;
close all;
fprintf('the classification with NFC is realizing\n');
performance=zeros(stepsize,1);
rr=single(zeros(fix(stepsize/25+1),4));
m=size(test,1);
[N,s2] = size(input);
clear data;
center=[];sigma_nf=[];targ=zeros(N,class);w=zeros(clustsize*class,class);w=sparse(w); 
sir=1;
for i=1:class
    [v,vv]=find(target_tr==i);
    temp=input(v,:);
    cent=mean(temp);
    [idc,cc]=kmeans(temp,clustsize,'MaxIter',10);
    center(sir:sir+clustsize-1,:)=cc;
    for j=1:clustsize
        ind=idc==j;
        sigma_nf(sir+j-1,:)=std(temp(ind,:));
        w(sir+j-1,i)=sum(ind)/size(v,1);
    end
    targ(v,i)=1;
    sir=sir+clustsize;
end
for i=1:s2
    ind=sigma_nf(:,i)<=0;
    sigma_nf(ind,i)=std(input(:,i));
end
[s1,s2]=size(center);
X = zeros(2*s1*s2,1);
X(1:s1*s2,1)=reshape(center',s1*s2,1);
X(s1*s2+1:2*s1*s2,1)=reshape(sigma_nf',s1*s2,1);

% Initial performance
[gX,out,w]=grad_anfis_aralik(input,center,sigma_nf,w,targ,class);
[tt,tp]=max(out');out_tr=tp';
init=sum(target_tr==out_tr)/N*100;   
perf=sum(sum((targ-out).^2))/N;
rr(1,:)=[0 init init perf];
fprintf('initial recognation rate= %g initial perform= %g',init,perf);
fprintf('\n'); 
result.center{1}=center;
result.sigma_nf{1}=sigma_nf;
result.w{1}=w;
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
num_X=1;
sir=2;
%The tarining of NFC with SCG is starting.
for epoch=1:stepsize        
    % If success is true, calculate second order information
    if (success == 1)
        sigmak = sigma/norm_dX;
        X_temp = X + sigmak*dX;
        center=(reshape(X_temp(1:s1*s2,1),s2,s1))';
        sigma_nf=(reshape(X_temp(s1*s2+1:2*s1*s2,1),s2,s1))';  
        for i=1:s2
            ind=sigma_nf(:,i)<=0;
            sigma_nf(ind,i)=std(input(:,i));
        end
        [gX_temp,out,w]=grad_anfis_aralik(input,center,sigma_nf,w,targ,class);        
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
        [rr(sir,:),recog_tr,recog_te,out_tr,out_te]=performance_measurement(epoch,perf_temp,out,target_tr,test,target_te,center,sigma_nf,w,class);          break;
        break;
    end
    
    % Calculate the comparison parameter  X_temp = X + alphak*dX;
    X_temp = X + alphak*dX;
    center=(reshape(X_temp(1:s1*s2,1),s2,s1))';
    sigma_nf=(reshape(X_temp(s1*s2+1:2*s1*s2,1),s2,s1))';
    for i=1:s2
        ind=sigma_nf(:,i)<=0;
        sigma_nf(ind,i)=std(input(:,i));
    end           
    out=output_anfis_aralik(input,center,sigma_nf,w,class);
    perf_temp=sum(sum((targ-out).^2))/N;    
    difk = 2*deltak*(perf - perf_temp)/(muk^2);
    
    % If difk >= 0 then a successful reduction in error can be made
    if (difk >= 0)
        gX_old = gX;
        X = X_temp;
        center=(reshape(X(1:s1*s2,1),s2,s1))';
        sigma_nf=(reshape(X(s1*s2+1:2*s1*s2,1),s2,s1))';      
        for i=1:s2
            ind=sigma_nf(:,i)<=0;
            sigma_nf(ind,i)=std(input(:,i));
        end
        [gX,out,w]=grad_anfis_aralik(input,center,sigma_nf,w,targ,class);        
        perf=sum(sum((targ-out).^2))/N;
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
        [rr(sir,:),recog_tr,recog_te,out_tr,out_te]=performance_measurement(epoch,perf_temp,out,target_tr,test,target_te,center,sigma_nf,w,class);                       
        result.center{sir}=center;
        result.sigma_nf{sir}=sigma_nf;
        result.w{sir}=w;
        sir=sir+1;
        if(rr(sir-2,4)==rr(sir-1,4))
            break;
        end
    end 
%gg(epoch,:)=gX;    
end
[~,best]=max(rr(:,2));
outputs.center=result.center{best};
outputs.sigma_nf=result.sigma_nf{best};
outputs.w=result.w{best};
labels.input=out_tr;
labels.test=out_te;
figure;plot(performance);
title('Performance evaluation');
xlabel('Epochs');
ylabel('RMSE value');
fismat=nfc_fis(double(input),double(target_tr),double(center),double(sigma_nf),class,clustsize);
ruleview(fismat);


%Functions
%**************************************************************************
%Calculation of gradients and NFC outputs
function [gX, out,w]=grad_anfis_aralik(input,center,sigma_nf,w,targ,class)
[s1,s2]=size(center);
N=size(input,1);
clust=s1/class;
if s2>1
    for i=1:s1
        temp=exp(-0.5*[(input-ones(N,1)*center(i,:)).^2]./(ones(N,1)*sigma_nf(i,:)).^2);    
        mem(:,i)=[prod([temp]')]';
    end
elseif s2==1
    for i=1:s1
        temp=exp(-0.5*[(input-ones(N,1)*center(i,:)).^2]./(ones(N,1)*sigma_nf(i,:)).^2);    
        mem(:,i)=temp;
    end
end
out_t=mem*w;
top=sum(out_t,2);
ind=top==0;
top(ind)=0.01;
out=out_t./(top*ones(1,class));
tempoc=zeros(s1*s2,1);tempos=zeros(s1*s2,1);
gX=zeros(2*s1*s2,1);
t1=-2*(targ-out);
sira=1;sir=1;
for k=1:class
    for j=1:s1/class        
        tempoc(sir:sir+s2-1)=tempoc(sir:sir+s2-1)+[(input-ones(N,1)*center(sira,:))./(ones(N,1)*sigma_nf(sira,:).^2)]'*[mem(:,sira).*t1(:,k).*(1-out(:,k))./top(:,1)*w(sira,k)];
        tempos(sir:sir+s2-1)=tempos(sir:sir+s2-1)+[(input-ones(N,1)*center(sira,:)).^2./(ones(N,1)*sigma_nf(sira,:).^3)]'*[mem(:,sira).*t1(:,k).*(1-out(:,k))./top(:,1)*w(sira,k)];        
        sir=sir+s2;sira=sira+1;
    end
end

gX=[tempoc;tempos]/N;


%**************************************************************************
%Calculation of only outputs
function [out]=output_anfis_aralik(input,center,sigma_nf,w,class)
[s1,s2]=size(center);
clustsize=s1/class;
N=size(input,1);
if s2>1
    for i=1:s1
        temp=exp(-0.5*[(input-ones(N,1)*center(i,:)).^2]./(ones(N,1)*sigma_nf(i,:)).^2);    
        mem(:,i)=[prod([temp]')]';
    end
elseif s2==1
    for i=1:s1
        temp=exp(-0.5*[(input-ones(N,1)*center(i,:)).^2]./(ones(N,1)*sigma_nf(i,:)).^2);    
        mem(:,i)=temp;
    end
end
out_t=mem*w;
top=sum(out_t,2);
ind=top==0;
top(ind)=0.01;
out=out_t./(top*ones(1,class));



%*************************************************************************
function [rr,recog_tr,recog_te,out_tr,out_te]=performance_measurement(epoch,perf_temp,out,target_tr,test,target_te,center,sigma_nf,w,class);
N=size(target_tr,1);
m=size(target_te,1);
[tt,tp]=max(out');out_tr=tp';
indx=(out_tr==target_tr);
recog_tr=sum(indx)/N*100;
output=output_anfis_aralik(test,center,sigma_nf,w,class);
[tt,tp]=max(output');out_te=tp';
indx=(out_te==target_te);
recog_te=sum(indx)/m*100;
fprintf('epoch %g   recog  %g  recog_test  %g  performans   %g\n',epoch,recog_tr,recog_te,perf_temp);
rr=[epoch recog_tr recog_te perf_temp];
   