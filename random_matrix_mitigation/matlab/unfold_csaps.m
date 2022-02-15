function [F,Cn,Cs]=unfold_csaps(E,npts) 
% unfolding of energy levels using CSAPS cubic spline of the cumulative
% distribution function.
% In:   E [nsamp,N] eigenvalues
%       npts [1,1]  number of equi-distant points   
% Out:  F [nsamp,N] unfolded eigenvalues
%       Cn cumulative distribution function of npts equi-distant levels.
%         Cn.x[npts,1] are the energy levels: 0<=Cn.x<=maxE-minE.
%         Cn.y[npts,1] are the values of Cn(x), 0<Cn.y<1.
%       Cs is smoothed cummulative distribution using cubic spline
%         Cs.x are the levels 0<=Cs.x<=maxE-minE
%         Cs.y are the splined values of Cs such that 0<Cs.y<1.

% Ref:  T.A. Brody et.al. Rev. Mod. Phys. {\bf 53} 386-480(1981).p.~391
%       T. Guhr et.al. Phys. Rep. {\bf 299}, 189-366(1998). p.~228.
% Note:  here we interpolate the smooth staircase function according Guhr
%       and not Brody (there is a slight difference in between).

if nargin==0
    help unfold_csaps
    return
end
[nsamp,N]=size(E);
if nsamp==1
    warning('unfold_csaps1::insufficient sampling');
end
if nargin<2 || isempty(npts)
    npts=ceil(max(20,N/10));
end
ev=E(:);
e0=min(ev);
ev=ev-e0;
% ev=sort(ev);
%% equi-distance bins
x0=linspace(0,max(ev),npts);
% mean level-spacing is
% d0=(max(ev)-min(ev))/N;    
P0=zeros(size(x0));         
%% cummulative distribution
for k=1:npts
    P0(k)=(1/nsamp/N)*length(find(ev<=x0(k)));
end
%% cubic spline of the cumulative distribution
val= fnplt(csaps(x0,P0)); % cubic spline (csaps(x0,P0,0)is linear)
x1=val(1,:); x1=x1(:);
y1=val(2,:); y1=y1(:);
%% eliminate repeated points
y1=max(y1,0);
y1=min(y1,1);
[x2,m1,n1] =unique(x1,'stable');
 y2=y1(m1);
%% interpolation such that F=C(E)
eu=interp1(x2,y2,ev);
%% arrange outputs
F=N*reshape(eu,size(E)); % eff. slope of CDF is 1.
Cn.x=x0';
Cn.y=P0';
Cs.x=x2;
Cs.y=y2;


%% smooth unfolded staircase function
if nargout==0 
    figure;
    plot(Cn.x,N*Cn.y,'b-',Cs.x,N*Cs.y,'g-');
    legend('equi-dist','smooth',2);
    hold on;
    plot(ev,N*eu,'r.');
    hold off;
    grid on;
    xlabel('\epsilon');
    ylabel('N*P(\epsilon)');
end





