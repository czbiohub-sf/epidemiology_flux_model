function G=mk_banded(K,N)
% K=0,(N) gives an [N,N] unit matrix,i.e., eye(N)
% K=1,(N) gives 3-diagonal matrix
% K=N-2   gives all ones except two zeros, at the corners (1,N),(N,1) 
% K=N-1   gives ones(N,N)
if K<0
    error('negative band');
end
if K>N-1
    error('band is too large');
end
% g=cell(1,2*K+1);
% q=0;
% for k=-K:1:K
%     q=q+1;
%     g{q}=diag(ones(1,N-abs(k)),k);
% end
% G=zeros(N,N);
% for k=1:2*K+1
%     G=G+g{k};
% end

zz=ones(N,N);
uu=triu(zz,K+1);
dd=tril(zz,-K-1);
G=zz-(uu+dd);
