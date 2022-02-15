% function [F,M]=load_flux_matrices()
% load the safeGraph flux matrices into the workspace
% In:	
% Out:	F[N,N,nsamp] a set of nsamp flux-matrices, each one of size [N,N]
%	    M[N,nsamp]   diagonal elements containing the population of communities 
% Note: input variables are prepared by set_param.m

% num:   date:
% 001    2019-01-01
% 335    2019-12-01
% 426    2020-03-01
% 450    2020-03-25
% 457    2020-04-01
% 459    2020-04-03
% 778    2021-02-16
ind1=426; % first day
ind2=778; % final day
%
nsamp=ind2-ind1+1;
F=zeros(N,N,nsamp);
M=zeros(N,nsamp);
tic;
s=0;
for q=ind1:ind2
    s=s+1;
    ff=load([dirName,'/',flist{q}]);
    F(:,:,s)=ff.F;
    M(:,s)=ff.Md;
    fprintf('%u\t',s);
    if mod(s,50)==0; fprintf('\n');end
end
fprintf('\n');
toc;
clear('ff','q','s');