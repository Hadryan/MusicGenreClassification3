function fismat=nfc_fis(input,target,center,sigma_nf,class,clustsize,pw)
% Initialize a FIS
fistype='sugeno';
[s1,s2]=size(center);
theStr = sprintf('%s%g%g',fistype,s2,1);
fismat = newfis(theStr, fistype);
if nargin<7
    pw=ones(s1,s2);
end

% Loop through and add inputs
for i = 1:1:s2
    
    fismat = addvar(fismat,'input',['in' num2str(i)],[min(input(:,i)) max(input(:,i))]);

    % Loop through and add mf's
    for j = 1:1:s1

        params = [center(j,i) sigma_nf(j,i) pw(j,i)];
        fismat = addmf(fismat,'input', i, ['in' num2str(i) 'cluster' num2str(j)], 'custmf1', params);

    end   
    
end


% Loop through and add outputs
fismat = addvar(fismat,'output',['out' num2str(1)],[min(target) max(target)]);
% Loop through and add mf's
sir=1;
for i=1:class
    for j = 1:1:clustsize
        params = [zeros(1,s2) i];
        fismat = addmf(fismat,'output', 1, ['out' num2str(i) 'cluster' num2str(j)], 'linear', params);
    end
end
        
% Create rules
ruleList = ones(s1, s2+1+2);
for i = 2:1:s1
    ruleList(i,1:s2+1) = i;    
end
fismat = addrule(fismat, ruleList);

% Set the input variable ranges
minX = min(input);
maxX = max(input);
ranges = [minX ; maxX]';
for i=1:s2
   fismat.input(i).range = ranges(i,:);
end

% Set the output variable ranges
minX = min(target);
maxX = max(target);
ranges = [minX ; maxX]';
for i=1:1
   fismat.output(i).range = ranges(i,:);
end


