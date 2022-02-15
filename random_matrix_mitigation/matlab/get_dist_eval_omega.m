% function [Es,Ts,Ls]=get_dist_eval_omega(F,B,rng,removedEdgeList)
% calculation of evals and total toll. The computation runs over selected values of the cutoff distances (in the range 1<=n<=ndst) and over all samples (i.e. the daily measured fluxes).  
% In: F[N,N,nsamp] flux matrices
%     B[N,N]       optional binary mask
%     removedEdgeLinks[1,ndst] ell array of links to be removed
% Out:Es[N,nsamp]  eigenvalues of all samples
%     Ts[1,nsamp]  total toll (=epidemic size) for all samples
%     Ls[2,nsamp]  error estimation of toll
% Use:flux2toll.m
% Note: The variable 'rng' that specifies the selected values of distances is set in the 
% main program. The results for each distance are saved in '/DIST/eval/nstr.mat' where 
% nstr is a numerical string in the range between 1 and ndst. 

Es=zeros(N,nsamp); % eigenvalues
Ts=zeros(1,nsamp); % omega for all samples
Ls=cell(2,nsamp);  % [L,err] of size [N,2] 
for n=rng
    if Nnzero(n)<2; continue;end
    for s=1:nsamp
        F1=F(:,:,s).*B;
        Ix=removedEdgeList{n}; %DIST is different than CENT
        F1(Ix)=0;
        E1=eig(F1);
        Es(:,s)=E1;
        [omega1,L1,err1]=flux2toll(F1*tolScale,M(:,s));
        Ts(s)=omega1;
        Ls{1,s}=L1;
        Ls{2,s}=err1;
        if mod(s,50)==0;fprintf('%u\t',s); end
    end
    %save([dir_fig,'/','eval','/',num2str(n,'%05u'),'.mat'],'n','Es','Ts','Ls');
    fprintf('\t%s\t%u\n','n=',n);
end