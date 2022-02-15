function [ps1,s1,d1,n1,wd,poi]=eig2spc(F,nbins,resmp)
% Calculation of level spacing distribution (for nearest levels) by histogram. 
% Use: wd_level_spacing.m
% Note: this is an early version using arbitrary number of histogram bins (not 
% recommended). See eval2ps.m for an improved version.
%
% In:  F [N,nsamp] spectrum
%      nbins[1,1]  number of histogram bins
%      resmp[1,1]  refining of the grid of WD and Poisson distributions
%                  (default: resmp=1 is same as the histogram)
% Out: ps1[nsamp*N,1] level distribution
%      s1[nsamp*N,1]  spacings
%      d1[1,1]       mean-level spacings
%      n1[1,1]       normalization (should be close to 1)
%      wd            struct for Wigner-Dyson pdf
%      poi           struct for Poisson pdf
if nargin<3 || isempty(resmp)
    resmp=1;
end
[N,nsamp]=size(F);
if nargin<2
    nbins=max(10,nsamp*N/20);
end
% 1st order difference along rows
if nsamp==1
    dF=diff(sort(F,'ascend'));
else
    dF=diff(F,1,1);
end
df=reshape(dF,1,(N-1)*nsamp);
if any(df<0)
    df(df<0)=nan;
    warning('negative spacings in F');
end
d1=nanmean(df); 
tol=1e-2;
if abs(d1-1)>tol
    warning('ps_dist1: mean level spacing differs from 1');
end
% function HIST is not recommended
% by Matlab
% [h1,s1]=hist(df,nbins);
% ds1=diff(s1); 
% ds1=[ds(1),ds];
% totArea=sum(ds.*h1);
% ps1=h1/totArea;
% n1=sum(ds.*ps1);
fig=figure;
set(fig,'visible','off');
hh=histogram(df,nbins,'normalization','pdf'); 
ps1=hh.Values;
s1=hh.BinWidth/2+hh.BinEdges;
ds1=diff(s1);
s1=s1(1:end-1);
n1=sum(ds1.*ps1); % sum (not trapz) because histogram is a bar-plot
close(fig);
%% Wigner-Dyson pdf (wigner surmise)
if resmp<=1
    s2=s1;
else
    s2=linspace(0,max(s1),ceil(resmp*nbins));
end
[ps2,n2]=wd_level_spacing(s2,d1,'goe');
wd.pval=ps2'; wd.sval=s2;wd.avg=d1;wd.nrm=n2;
%% Poisson pdf
s3=s2;
ps3=exp(-s3/d1)/d1;
ds3=diff(s3); 
ds3=[ds3(1),ds3];
n3=trapz(ds3.*ps3);
poi.pval=ps3; poi.sval=s3;poi.avg=d1;poi.nrm=n3;

