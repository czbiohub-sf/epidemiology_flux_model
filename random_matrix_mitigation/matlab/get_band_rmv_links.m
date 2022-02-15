%function [removedEdgeList,Nnzero]=get_band_rmv_links(ndst)
% determine a set of links to be removed according to bandwidth
% In:	ndst[1,1] range of bandwidths, i.e., number of non-zero diagonals 
% Out:  removedEdgeList[1,ndst] cell array of links to be removed at step n (1<=n<=ndst).
%       Nnzero[1,ndst] number of non-zero elements at step n.
% Use:  mk_banded.m
% Note: The results are saved in file '/BAND/rmv_band_toll.mat'. In the case of BAND, 
% removedEdgeList{n} contains a list of removed links up to step n (n included).

ndst=N-1;
removedEdgeList=cell(1,ndst);
numCuts=zeros(1,ndst);
%% make a banded matrix
for n=1:ndst
    B=mk_banded(n,N);
    Ix=find(B==0);
    Ix0=Ix;
    removedEdgeList{n}=Ix;
    numCuts(n)=length(Ix);
    fprintf('%u\t',n);
    if mod(n,50)==0; fprintf('\n');end
end
Nnzero=N^2-numCuts;
save([dir_fig,'/','rmv_band_toll.mat'],'ndst','Nnzero','removedEdgeList','-v7.3');


% figure;
% plot(dd/1000,Nnzero,dd/1000,numCuts); 
% grid on;
% legend('nnz','numCuts');
% xlabel('d [km]');
% ylabel('#nnz');
% return
clear('s','n');