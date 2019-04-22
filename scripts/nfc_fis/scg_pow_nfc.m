function [fismat,outputs,recog_tr,recog_te,labels,performance]=scg_pow_nfc(input,target_tr,test,target_te,stepsize,class,clustsize);
% In this program, the neuro-fuzzy classifier parameters are adapted by Scaled conjugate gradient method.
%Also, the power values are applied to the fuzzy sets and adapted with SCG
%method.
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
%outputs.pw[s1,s2]:The power values of Gaussian functions 
%fismat: Demonstration of NFC in fuzzy viewer.
%       Written by Dr. Bayram Cetiþli Suleyman Demirel University Computer
%       Engineeering Isparta Turkey
close all;
performance=single(zeros(stepsize,1));
warning off;
fprintf('the classification with NFC_LH is realizing\n');
rr=single(zeros(stepsize/25,4));
m=size(test,1);
[N,s2] = size(input);
input=single(input);test=single(test);
target_tr=uint8(target_tr);target_te=uint8(target_te);
center=single(zeros(clustsize*class,s2));sigma_nf=single(zeros(clustsize*class,s2));
targ=uint8(zeros(N,class));w=single(zeros(clustsize*class,class));
sir=1;
for i=1:class
    [v,vv]=find(target_tr==i);
    temp=input(v,:);
    cent=mean(temp);
    [idc,cc]=kmeans(temp,clustsize,'MaxIter',s2);
    center(sir:sir+clustsize-1,:)=cc;
    for j=1:clustsize
        ind=idc==j;
        sigma_nf(sir+j-1,:)=std(temp(ind,:));
        w(sir+j-1,i)=sum(ind)/size(v,1);
    end
    targ(v,i)=1;   
    sir=sir+clustsize;
end
clear cent m1 v1 w1
for i=1:s2
    ind=sigma_nf(:,i)<=0;
    sigma_nf(ind,i)=std(input(:,i));
end
clear ind
[s1,s2]=size(center);
pw=single(1*ones(s1,s2));
X = single(zeros(3*s1*s2,1));
X(1:s1*s2,1)=reshape(center',s1*s2,1);
X(s1*s2+1:2*s1*s2,1)=reshape(sigma_nf',s1*s2,1);
X(2*s1*s2+1:end,1)=reshape(pw',s1*s2,1);
% Initial performance
[gX,out,w]=grad_anfis_aralik(input,center,sigma_nf,pw,w,targ,class);
ind=isnan(gX);
gX(ind)=0.0001;
[tt,tp]=max(out');out_tr=uint8(tp');
init=sum(target_tr==out_tr)/N*100;   
perf=sum(sum((single(targ)-out).^2))/N;
rr(1,:)=[1 init init perf];
fprintf('initial recognation rate= %g initial perform= %g',init,perf);
fprintf('\n'); 

result.center{1}=center;
result.sigma_nf{1}=sigma_nf;
result.w{1}=w;
result.pw{1}=pw;
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

%The tarining of NFC with SCG is starting.
sir=2;
for epoch=1:stepsize        
    % If success is true, calculate second order information
    if (success == 1)
        if norm_dX~=0
            sigmak = sigma/norm_dX;
        else
            sigmak=sigma;
        end
        X_temp = X + sigmak*dX;
        center=(reshape(X_temp(1:s1*s2,1),s2,s1))';
        sigma_nf=(reshape(X_temp(s1*s2+1:2*s1*s2,1),s2,s1))';                  
        pw=(reshape(X_temp(2*s1*s2+1:end,1),s2,s1))';
        for i=1:s2
            ind=sigma_nf(:,i)<=0;
            sigma_nf(ind,i)=std(input(:,i));
        end        
        [gX_temp,out,w]=grad_anfis_aralik(input,center,sigma_nf,pw,w,targ,class);
        ind=isnan(gX_temp);
        gX_temp(ind)=0.0001;
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
        [rr(sir,:),recog_tr,recog_te,out_tr,out_te]=performance_measurement(epoch,perf_temp,out,target_tr,test,target_te,center,sigma_nf,pw,w,class);        
        break;
    end
    
    % Calculate the comparison parameter  X_temp = X + alphak*dX;
    X_temp = X + alphak*dX;
    center=(reshape(X_temp(1:s1*s2,1),s2,s1))';
    sigma_nf=(reshape(X_temp(s1*s2+1:2*s1*s2,1),s2,s1))';       
    pw=(reshape(X_temp(2*s1*s2+1:end,1),s2,s1))';
    for i=1:s2
        ind=sigma_nf(:,i)<=0;
        sigma_nf(ind,i)=std(input(:,i));
    end
    
    out=output_anfis_aralik(input,center,sigma_nf,pw,w,class);
    perf_temp=sum(sum((single(targ)-out).^2))/N;    
    difk = 2*deltak*(perf - perf_temp)/(muk^2);
    
    % If difk >= 0 then a successful reduction in error can be made
    if (difk >= 0)
        gX_old = gX;
        X = X_temp;
        center=(reshape(X(1:s1*s2,1),s2,s1))';
        sigma_nf=(reshape(X(s1*s2+1:2*s1*s2,1),s2,s1))';      
        pw=(reshape(X_temp(2*s1*s2+1:end,1),s2,s1))';   
        for i=1:s2
            ind=sigma_nf(:,i)<=0;
            sigma_nf(ind,i)=std(input(:,i));
        end
        
        [gX,out,w]=grad_anfis_aralik(input,center,sigma_nf,pw,w,targ,class);        
        ind=isnan(gX);
        gX(ind)=0.0001;
        perf=sum(sum((single(targ)-out).^2))/N;
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
        lambdak = lambdak + deltak*(1 - difk)/nrmsqr_dX;
    end   
    performance(epoch,1)=perf;
    if rem(epoch,25)==0 | rem(epoch,stepsize)==0
        [rr(sir,:),recog_tr,recog_te,out_tr,out_te]=performance_measurement(epoch,perf_temp,out,target_tr,test,target_te,center,sigma_nf,pw,w,class);        
        result.center{sir}=center;
        result.sigma_nf{sir}=sigma_nf;
        result.w{sir}=w;
        result.pw{sir}=pw;
        if rr(sir,4)==rr(sir-1,4)
             break;
         end
        sir=sir+1;     
    end  
    if epoch==stepsize
        ind=find(pw<0);
        pw(ind)=0;
    end
   % gg(epoch,:)=gX;    

end
[~,best]=max(rr(:,2));
outputs.center=result.center{best};
outputs.sigma_nf=result.sigma_nf{best};
outputs.w=result.w{best};
outputs.pw=result.pw{best};
labels.input=out_tr;
labels.test=out_te;
figure;plot(performance);
title('Performance evaluation');
xlabel('Epochs');
ylabel('RMSE value');
fismat=nfc_fis(double(input),double(target_tr),double(center),double(sigma_nf),class,clustsize,double(pw));
ruleview(fismat);


%Functions
%**************************************************************************
%Calculation of gradients and NFC outputs
function [gX, out,w]=grad_anfis_aralik(input,center,sigma_nf,pw,w,targ,class)
[s1,s2]=size(center);
N=size(input,1);temp=single(zeros(N,s2));mem=single(zeros(N,s1));
for i=1:s1
    temp=exp(-0.5*[(input-ones(N,1)*center(i,:)).^2]./(ones(N,1)*sigma_nf(i,:)).^2);
    mem(:,i)=[prod([temp.^(ones(N,1)*pw(i,:))]')]';
end
ind=isinf(mem);
mem(ind)=1;
ind=isnan(mem);
mem(ind)=1;
ind=sum(mem,2)==0;
mem(ind,1)=1;
out_t=mem*w;
top=single(sum(out_t,2));
ind=top==0;
top(ind)=0.01;
out=out_t./(top*ones(1,class));
tempoc=single(zeros(s1*s2,1));tempos=single(zeros(s1*s2,1));
tempopw=single(zeros(s1*s2,1));gX=single(zeros(3*s1*s2,1));
t1=-2*(single(targ)-out);
sira=1;sir=1;temp=single(zeros(N,s2));
for k=1:class    
    for j=1:s1/class
        temp=exp(-0.5*[(input-ones(N,1)*center(sira,:)).^2]./(ones(N,1)*sigma_nf(sira,:)).^2);
        [v,vv]=find(temp==0);
        temp(v,vv)=0.000001;
        tempoc(sir:sir+s2-1)=tempoc(sir:sir+s2-1)+([(input-ones(N,1)*center(sira,:))./(ones(N,1)*sigma_nf(sira,:).^2)]'*[mem(:,sira).*t1(:,k).*(1-out(:,k))./top(:,1)*w(sira,k)]).*pw(sira,:)';
        tempos(sir:sir+s2-1)=tempos(sir:sir+s2-1)+([(input-ones(N,1)*center(sira,:)).^2./(ones(N,1)*sigma_nf(sira,:).^3)]'*[mem(:,sira).*t1(:,k).*(1-out(:,k))./top(:,1)*w(sira,k)]).*pw(sira,:)';
        tempopw(sir:sir+s2-1)=tempopw(sir:sir+s2-1)+log(temp')*[t1(:,k).*mem(:,sira).*(1-out(:,k))./top(:,1)]*w(sira,k);
        sir=sir+s2;sira=sira+1;
    end
end
gX=[tempoc;tempos;tempopw]/N;


%**************************************************************************
%Calculation of only outputs
function [out]=output_anfis_aralik(input,center,sigma_nf,pw,w,class)
[s1,s2]=size(center);
clustsize=s1/class;
N=size(input,1);
temp=single(zeros(N,s2));mem=single(zeros(N,s1));
for i=1:s1
    temp=exp(-0.5*[(input-ones(N,1)*center(i,:)).^2]./(ones(N,1)*sigma_nf(i,:)).^2);   
    mem(:,i)=[prod([temp.^(ones(N,1)*pw(i,:))]')]';
end
ind=isinf(mem);
mem(ind)=1;
ind=isnan(mem);
mem(ind)=1;
ind=sum(mem,2)==0;
mem(ind,1)=1;
out_t=mem*w;
top=single(sum(out_t,2));
ind=top==0;
top(ind)=0.01;
out=out_t./(top*ones(1,class));



%*************************************************************************
function [rr,recog_tr,recog_te,out_tr,out_te]=performance_measurement(epoch,perf,out,target_tr,test,target_te,center,sigma_nf,pw,w,class);
N=size(target_tr,1);
m=size(target_te,1);
[tt,tp]=max(out');out_tr=tp';
indx=(out_tr==target_tr);
recog_tr=sum(indx)/N*100;
output=output_anfis_aralik(test,center,sigma_nf,pw,w,class);
[tt,tp]=max(output');out_te=tp';
indx=(out_te==target_te);
recog_te=sum(indx)/m*100;
fprintf('epoch %g   recog  %g  recog_test  %g  performans   %g\n',epoch,recog_tr,recog_te,perf);
rr=[epoch recog_tr recog_te perf];
   