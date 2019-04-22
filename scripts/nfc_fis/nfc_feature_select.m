function [fismat,feature,outputs,recog_tr,recog_te,labels,performance]=nfc_feature_select(input,target_tr,test,target_te,stepsize,class);

% In this program, the performance of SCG_NFC is improved using power.
%According to the power values of features, some of the features are
%accepted to train or rejected. Note that the centers and widths of Gaussian functions are not trained during the NFC training.

%INPUTS
%input[N,s2]: training data
%target[N,1]: the target values of training data
%test[m,s2]: test data
%hedef[m,1]: the target values of test data
%stepsize: The maximum iteration number
%class: Number of classes
%clustsize: Number of cluster of each class
%OUTPUTS
%center[s1,s2]: The center values of Gaussian functions of i-th rule and j-th feature 
%sigma_nf[s1,s2]: The width values of Gaussian functions of i-th rule and j-th feature 
%recog: The recognition rate of training data 
%recog_test: The recognition rate of test data 
%out_t[N,1]: The actual class labels of training data obtained from NFC 
%output_t[m,1]: The actual class labels of test data obtained from NFC
%performans: meas square error of training data 
%pw[s1,s2]:The power values of Gaussian functions of i-th rule and j-th feature 
%warning off;

%bu fonksiyon sadece kuvvetleri kullanarak featurelarýn nasýl seçileceðini
%göstermektedir.
close all;
performance=zeros(stepsize,1);
sir=2;
rr=single(zeros(stepsize/25+1,4));
m=size(test,1);
[N,s2] = size(input);
center=zeros(class,s2);sigma_nf=zeros(class,s2);
targ=zeros(N,class);w=zeros(class,class);
for i=1:class
    [v,vv]=find(target_tr==i);
    temp=input(v,:);    
    center(i,:)=mean(temp);
    sigma_nf(i,:)=std(temp);
    w(i,i)=1;
    targ(v,i)=1;    
end
[s1,s2]=size(center);
pw=0.0001*ones(s1,s2);
X = zeros(s1*s2,1);
X(1:end,1)=reshape(pw',s1*s2,1);
% Initial performance
[gX,out]=grad_anfis_aralik(input,center,sigma_nf,pw,w,targ,class);
[tt,tp]=max(out');out_t=tp';
init=sum(target_tr==out_t)/N*100;   
perf=sum(sum((targ-out).^2))/N;
perf_b=perf;
fprintf('initial recognation rate= %g initial perform= %g',init,perf);
rr(1,:)=[0 init init perf];
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
num_X=1;

%SCG ile parametrelerin uyarlanarak e?itimin yap?lmas?
tic;
for epoch=1:stepsize        
    % If success is true, calculate second order information
    if (success == 1)
        sigmak = sigma/norm_dX;
        X_temp = X + sigmak*dX;                       
        pw=(reshape(X_temp(1:end,1),s2,s1))';
        [gX_temp,out]=grad_anfis_aralik(input,center,sigma_nf,pw,w,targ,class);
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
    pw=(reshape(X_temp(1:end,1),s2,s1))';
    out=output_anfis_aralik(input,center,sigma_nf,pw,w,class);
    perf_temp=sum(sum((targ-out).^2))/N;    
    difk = 2*deltak*(perf - perf_temp)/(muk^2);
    
    % If difk >= 0 then a successful reduction in error can be made
    if (difk >= 0)
        gX_old = gX;
        X = X_temp;
        pw=(reshape(X_temp(1:end,1),s2,s1))';            
        [gX,out]=grad_anfis_aralik(input,center,sigma_nf,pw,w,targ,class);        
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
        lambdak = lambdak + deltak*(1 - difk)/nrmsqr_dX;
    end   
    performance(epoch,1)=perf;
    if rem(epoch,25)==0 | rem(epoch,stepsize)==0
        [rr(sir,:),recog_tr,recog_te,out_tr,out_te]=performance_measurement(epoch,perf_temp,out,target_tr,test,target_te,center,sigma_nf,pw,w,class);        
        sir=sir+1;
        if(rr(sir-2,4)==rr(sir-1,4))
            break;
        end
    end   
    [v,vv]=find(pw<=0);
    [v1,vv1]=find(pw>=1);
    d1=(pw<0);d2=(pw>1);
    if (size(v,1)>1 | size(v1,1)>1)
       pw(d1)=0;
       pw(d2)=1;       
    end
    
end
pw=(reshape(X(1:s1*s2,1),s2,s1))';
ind=pw<0;
pw(ind)=0;
ind=pw>1;
pw(ind)=1;
[members]=membership_f([zeros(1,s2);ones(1,s2)],center,sigma_nf,pw,w,class);
PW=sum(pw,1);
[v,vv]=sort(PW,'descend');
figure;bar(vv,v,0.1);
title('Feature selection criteria');
xlabel('features');
ylabel('total linguistic hedge values');
feature.index=vv;
feature.power=v;
[v,vv]=sort(PW,'descend');
if size(vv,2)>round(s2/2)
    feature.selected=vv(1:round(s2/2));
else
    feature.selected=vv;
end
ind=pw==0;
index=sum(ind,1)>=class-1;
[~,vv]=find(index==1);
feature.rejected=vv;
temp=[];
for i=1:size(feature.selected,2)
    for j=1:size(feature.rejected,2)
        if feature.selected(i)==feature.rejected(j)
            temp=[temp;i];
        end
    end
end
feature.selected(temp)=[];
outputs.center=center;
outputs.sigma_nf=sigma_nf;
outputs.pw=pw;
outputs.w=w;
outputs.mf=members;
labels.input=out_tr;
labels.test=out_te;
fismat=nfc_fis(double(input),double(target_tr),double(center),double(sigma_nf),class,1,double(pw));
ruleview(fismat);
figure;plot(performance);
title('Performance evaluation');
xlabel('Epochs');
ylabel('RMSE value');


%Functions
%**************************************************************************
%Calculation of gradients and NFC outputs
function [gX, out]=grad_anfis_aralik(input,center,sigma_nf,pw,w,targ,class)
[s1,s2]=size(center);
N=size(input,1);
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
top=sum(out_t,2);
out=out_t./(top*ones(1,class));
tempopw=zeros(s1*s2,1);gX=zeros(s1*s2,1);
t1=-2*(targ-out);
sira=1;sir=1;
for k=1:class
    temp=zeros(N,s2);
    for j=1:s1/class
        temp(:,:)=exp(-0.5*[(input-ones(N,1)*center(sira,:)).^2]./(ones(N,1)*sigma_nf(sira,:)).^2);
        [v,vv]=find(temp==0);
        temp(v,vv)=0.000001;
        tempopw(sir:sir+s2-1)=tempopw(sir:sir+s2-1)+log(temp')*[t1(:,k).*mem(:,sira).*(1-out(:,k))./top(:,1)]*w(sira,k);
        sir=sir+s2;sira=sira+1;
    end
end
gX=[tempopw]/N;

%**************************************************************************
%Calculation of NFC outputs
function [out,mem]=output_anfis_aralik(input,center,sigma_nf,pw,w,class)
[s1,s2]=size(center);
clustsize=s1/class;
N=size(input,1);
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
top=sum(out_t,2);
out=out_t./(top*ones(1,class));



%*************************************************************************
%Performance measurement
function [rr,recog_tr,recog_te,out_tr,out_te]=performance_measurement(epoch,perf_temp,out,target_tr,test,target_te,center,sigma_nf,pw,w,class);
N=size(target_tr,1);
m=size(target_te,1);
[tt,tp]=max(out');out_tr=tp';
indx=(out_tr==target_tr);
recog_tr=sum(indx)/N*100;
output=output_anfis_aralik(test,center,sigma_nf,pw,w,class);
[tt,tp]=max(output');out_te=tp';
indx=(out_te==target_te);
recog_te=sum(indx)/m*100;
fprintf('epoch %g   recog_train  %g  recog_test  %g  performance   %g\n',epoch,recog_tr,recog_te,perf_temp);
rr=[epoch recog_tr recog_te perf_temp];


%*************************************************************************
%calculation of boundary conditions
function [members]=membership_f(input,center,sigma_nf,pw,w,class)
[s1,s2]=size(center);
clustsize=s1/class;
N=size(input,1);
members=[];
for i=1:s1
    temp=(exp(-0.5*[(input-ones(N,1)*center(i,:)).^2]./(ones(N,1)*sigma_nf(i,:)).^2)).^(ones(N,1)*pw(i,:));   
    members=[members;temp];
end
    