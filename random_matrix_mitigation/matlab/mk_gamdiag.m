function G=mk_gamdiag(F)
% In:	F[N,N,nsamp] a set of [N,N] flux matrices 
% Out:  G[N,N,nsamp] same as F with the diagonals replaced by Gamma random variables. The % Note: The mean of each r.v. equals to the average of F over rows (or cols) and the std is half of the mean.

tet=0.5;
fac=1e8;
[N1,N2,nsamp]=size(F);
if N1~=N2
    error('square matrix expected');
end
N=N1; clear('N2');

M2=mean(F,3);
M1=mean(M2,2);
K=M1/tet;
kk=fac*repmat(K,[1,nsamp]);
tt=repmat(tet,[N,nsamp]);
%for reproducibility
stream = RandStream.getGlobalStream;
savedState = stream.State; %stream.State = savedState; 
R=gamrnd(kk,tt);

G=F;
for n=1:N
    for s=1:nsamp
        G(n,n,s)=R(n,s)/fac;
    end
end
