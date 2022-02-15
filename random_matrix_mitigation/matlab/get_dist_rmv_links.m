%function [dd,removedEdgeList,Nnzero]=get_dist_rmv_links(dst)
% determine a set of links to be removed according to distance matrix given in dst.mat
% In:	dst[N,N] symmetric matrix of distances between communities [meter]
% Out:  dd [ndst,1]  cut-off distances [meter]
%       removedEdgeList[1,ndst] cell array of links of lengths larger than dd(n) 
%                 where the iterator n is in the range (1<=n<=ndst).
%       Nnzero[1,ndst] number of non-zero elements at step n.
% Note: The results are saved in file '/DIST/rmv_dist_toll.mat'. In the case of DIST, 
% removedEdgeList{n} contains a list of removed links up to step n (n included).

dst=load([dir_fig,'/','dst.mat']);
dst=dst.dst; 
[N1,N2]=size(dst);
if N1~=N || N2 ~=N
    error('inconsistent size of distance matrix');
end   
clear('N1','N2');
dd=(100:10:12000)*1e3; %[m] Note: min(dst)=0, max(dst)=11784[km]
% correction of minimal range (where 'KL' has systematic errors) 
firstInd=8;
% correction of maximal range (see 'no-links' below)
lastInd=1169-(firstInd-1);     
dd([1:firstInd-1,lastInd+1:end])=[];
dd=(170:200:11700)*1e3;
ndst=length(dd);


% allocation
removedEdgeList=cell(1,ndst);
numCuts=zeros(1,ndst);
Ix0=[];
for n=1:ndst
    Ix=find(dst>dd(n));
    if isempty(Ix)
        fprintf('\n%s\t%u\t%f\n','no-links',n,dd(n));
        removedEdgeList{n}=nan;
        continue
    else
        if isempty(setxor(Ix,Ix0))
            fprintf('\n%s\t%u\t%f\n','same-set',n,dd(n)); 
        end
    end
    Ix0=Ix;
    removedEdgeList{n}=Ix;
    numCuts(n)=length(Ix);
    fprintf('%u\t',n);
    if mod(n,50)==0; fprintf('\n');end
end
Nnzero=N^2-numCuts;
save([dir_fig,'/','rmv_dist_toll.mat'],'dd','ndst','Nnzero','removedEdgeList');

% figure;
% plot(dd/1000,Nnzero,dd/1000,numCuts); 
% grid on;
% legend('nnz','numCuts');
% xlabel('d [km]');
% ylabel('#nnz');
% return
clear('s','n');