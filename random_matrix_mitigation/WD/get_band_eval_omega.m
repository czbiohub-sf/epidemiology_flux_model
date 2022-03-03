% function [Es,Ts,Ls]=get_band_eval_omega(F,rng)
% calculation of evals and total toll. The computation runs over bandwidths 1<=n<=N-1 and % over all samples (i.e. the daily measured fluxes).  
% In: F[N,N,nsamp] flux matrices
%
% Out:Es[N,nsamp]  eigenvalues of all samples
%     Ts[1,nsamp]  total toll (=epidemic size) for all samples
%     Ls[2,nsamp]  error estimation of toll
% Use: mk_banded, flux2toll.m
% Note: The variable 'rng' that specifies the selected values of bandwidth (i.e. number of % is set in the main program. The results for each bandwidth are saved in '/BAND/eval/
% nstr.mat' where nstr is a numerical string in the range between 1 and ndst. 


Es=zeros(N,nsamp); % eigenvalues
Ts=zeros(1,nsamp);   % omega for all samples
Ls=cell(2,nsamp);  % [L,err] of size [N,2] 
for n=rng
    %Ix=removedEdgeList{n}; % different than DIST
    B=mk_banded(n,N); % i
    for s=1:nsamp
        F1=F(:,:,s).*B;
        %F1(Ix)=0;
        if nnz(F1)<2; continue;end
        E1=eig(F1);
        Es(:,s)=E1;
        [omega1,L1,err1]=flux2toll(F1*tolScale,M(:,s));
        Ts(s)=omega1;
        Ls{1,s}=L1;
        Ls{2,s}=err1;
        if mod(s,50)==0;fprintf('%u\t',s); end
    end
    save([dir_fig,'/','eval','/',num2str(n,'%05u'),'.mat'],'n','Es','Ts','Ls','B');
    fprintf('\t%s\t%u\n','n=',n);
end