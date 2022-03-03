%function [removedEdgeList,Nnzero, ndst]=get_cent_rmv_links(grp)
% determine a set of links to be removed according to betweenness edge-centrality.
% In:	grp[1,1] initial number of links to be removed in one chunk. Default grp=10,0000.
%                this value decreases adaptively.
% Out:  removedEdgeList[1,ndst] cell array of links to be removed at step n (1<=n<=ndst)
%       Nnzero[1,ndst] number of non-zero elements at step n.
%       ndst[1,1] number of actual iterations.
% Use:  edge_centrality.m  (MalabBGL by D. Geich, 2022) 
% Note: The results are saved in file '/BAND/rmv_band_toll.mat'. In the case of CENT, 
% removedEdgeList{n} contains a list of removed AT step n (n included) and, as opposed to % DIST and BAND, not cumulative (cumulative=up to step n).

% parameters
maxIter=7000;   % maximum number of iterations
grp=10000;      % initial number of edges to be removed in one chunk
% allocation
removedEdgeList=cell(1,maxIter);
Nnzero=zeros(1,maxIter);
% params
H=mean(F,3);
tolH=0.90;
n0=nnz(H);
T0=15; %[sec]
tolT=0.95;
N=size(H,1);
for n=1:maxIter
    n1=nnz(H);
    H1=H+eye(N);
    if nnz(H1)<=N %2*N
        break
    end
    if (n1/n0)<=tolH
        grp=ceil(grp*tolH);
        n0=n1;
    end
    Nnzero(n)=n1;
    Hs=sparse(H);
    tic;
    [Cv,Ce] = betweenness_centrality(Hs);
    T=toc;
    if T/T0<tolT
        grp=ceil(grp*tolT);
        T0=T;
    end
    [i,j,e]=find(triu(Ce)); % using symmetry
    [~,ind]=sort(e,1,'descend');
    if length(ind)>grp
       ind=ind(1:grp);
    end
    Ix1=zeros(size(ind));
    Ix2=zeros(size(ind));
    for k=1:length(ind)
        m=ind(k);
        Ix1(k)=sub2ind(size(H),i(m),j(m));
        Ix2(k)=sub2ind(size(H),j(m),i(m));
    end
    removedEdgeList{n}=[Ix1;Ix2];
    H(removedEdgeList{n})=0;
    fprintf('%u\t%u\t%u\t%f\n',n,grp,Nnzero(n),T);
end
%save([dir_fig,'/','rmv_cent_toll.mat'],'ndst','Nnzero','removedEdgeList');
ndst=n+1;
tmp=cell(1,ndst);
tmp{1}=[];
for k=2:ndst
    tmp{k}=removedEdgeList{k-1};
end
removedEdgeList=tmp;
clear('tmp');
Nnzero=Nnzero(1:ndst);
save([dir_fig,'/','rmv_cent_toll.mat'],'ndst','Nnzero','removedEdgeList');
clear('s','n');