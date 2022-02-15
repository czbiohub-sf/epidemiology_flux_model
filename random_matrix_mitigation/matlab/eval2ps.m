function [KL,p1,s1,p2,p3,d1,d2,nrm,nrm2,nrm3,Gs]=eval2ps(Es)
% Calculation of the level-spacing distribution and KL divergence from sampled energy 
% levels using histogram with adaptive number of bins. 
% Use: ecdf, ecdfhist, wd_level_spacing.m
% Note: this is improved version as compared to eig2spc.m

% for Use empirical cumulative dist to determine 
% In:	Es[N,nsamp] spectra of nsamp matrices of size [N,N]
% Out:  Gs[N,nsamp] smoothed and interpolated empirical cumulative distribution function
%                   of the energy levels.
%       p1[N,nsamp] level spacing distribution (nearest levels)
%       s1[N,nsamp] spacings
%       nrm[1,1]    normalization of p1 (expected to be close to 1)
%       d1[1,1]     empirical mean level spacing (estimated from Gs)
%       d2[1,1]     mean level spacing estimated from [p1,s1]
%       p2[N*samp,1]  Wigner-Dyson distribution for GOE.
%       nrm2[1,1]     normalization (expected to be close to 1)
%       p3[N*samp,1]  Poisson distribution p3=exp(-s3).
%       nrm3[1,1]     normalization (expected to be close to 1)
%       KL[1,1]      KL divergence D[p1||Poisson]


tol=1e-3;
maxIter=15;
fac=1.1;    % increase of nbin at each iteration
spn=15;     % span for smoothing

[N,nsamp]=size(Es);
ev=Es(:);e0=min(ev);ev=ev-e0;
[p0,x0]=ecdf(ev);
%figure; plot(x0,p0,'.-'); grid on;
val= fnplt(csaps(x0,p0)); % cubic spline (csaps(x0,p0,0)is linear)
x1=val(1,:); x1=x1(:);
y1=val(2,:); y1=y1(:);
y1=max(y1,0); 
y1=min(y1,1);
%figure; plot(x0,p0,'-',x1,y1,'-'); grid on;
[x2,m1,~] =unique(x1,'stable');
y2=y1(m1);
eu=interp1(x2,y2,ev);
Gs=N*reshape(eu,[N,nsamp]); % eff. slope of CDF is 1.
%figure; plot(1:N,Gs(:,1),'.-',1:N,Gs(:,nsamp),'.-'); grid on
Fs=diff(Gs,1,1);
ff=reshape(Fs,1,(N-1)*nsamp);
d1=mean(ff);
[gg,xx] = ecdf(ff);
%figure; plot(xx,gg); grid on
nbin=ceil(sqrt(length(ff))/2);
KL0=100;KL=10;
iter=0;
while abs(KL-KL0)>tol || iter<maxIter
    iter=iter+1;
    [p1,s1] = ecdfhist(gg,xx,nbin);
    %figure; plot(s1,p1); grid on;
    p1(p1<=0)=eps;
    nrm=trapz(s1,p1);
    p1=p1/nrm;
    p1=smooth(p1,spn);p1=p1.';
    d2=trapz(s1,s1.*p1);
    KL0=KL;
    KL=trapz(s1,p1.*(log(p1)+s1));
    nbin=ceil(fac*nbin);
end
[p2,nrm2]=wd_level_spacing(s1,d2,'GOE');
[p3,nrm3]=wd_level_spacing(s1,d2,'POISSON');
% if abs(nrm2-1)>1e-3 || abs(nrm3-1)>1e-3
%     warning('norm=1 is expected');
% end
% H=-trapz(s1,p1.*(log(p1))); % entropy