function out = custmf1(input, params)
N=size(input,1);
out=(exp(-0.5*(input-ones(N,1)*params(1)).^2./(ones(N,1)*params(2)).^2)).^params(3);
