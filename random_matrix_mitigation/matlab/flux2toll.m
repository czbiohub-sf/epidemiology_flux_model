function [omega,Lfin,Eerr]=flux2toll(F,M)
% calculation of total toll (epidemic size)
% In:   F[n,n]     flux matrix for n communities
% Out:  Lfin[n,1]  final time integrated infectivities $\mu_n(t\rightarrow\infty)$
%       Eerr[n,1]  final iteration errors
%       omega[1,1] total toll 
% Ref: Supplementary materials Eqns. (S13)-(S15).

%See: RMT-of-epidemics/Matlab/Matlab_toll/chk_goe4.m   
[n1,n2]=size(F);
if n1~=n2
    error('square matrix is expected');
end
n=n1; 
clear('n2');
% relative population
if nargin<2 || isempty(M)
    p=ones(n,1)/n;     
else
    p=M/sum(M);
end
ep=(1e-4)*ones(n,1);
gam=0.125; % gamma=0.1 (glt)
tolL=1e-6;
maxIter=1000;

L0=ones(n,1)/gam;  % initial solution
tmp=zeros(n,maxIter);
Lfin=zeros(n,1);
dL=ones(n,1);
L1=L0;
iter=0;
while abs(max(dL))>tolL && iter<=maxIter
        iter=iter+1;
        tmp(:,iter)=L1;
        q=p.*(1-(1-ep).*exp(-L1));
        gamL=F*q;
        L2=gamL/gam;
        dL=L2-L1;
        L1=L2;
end
Lfin(:)=tmp(:,iter);
omega=1-sum(p.*(1-ep).*exp(-Lfin(:)),1);
% chk convergence
v=p.*(1-(1-ep).*exp(-Lfin));
Eerr=F*v-gam*Lfin;
%typErr=mean(abs(Eerr));
