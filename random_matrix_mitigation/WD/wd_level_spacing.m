function [p,nrm]=wd_level_spacing(s,d,typ)
% Wigner_Dyson level-spacing distribution (LSD)
% using Wigner's surmise.
%Ref: T. Guhr et.al. Phys. Rep. {\bf 299}, 189-366(1998).  

%In:    s[N,1]  nearest level spacing
%       d[1,1]  mean-level spacing. default=1
%       typ     type: ,GUE,GOE, Poisson
%Out:   p[N,1]  p(s)
%       nrm[1,1] 


if nargin==0
    help wd_level_spacing;
    return
end
if nargin<3
    typ=poisson;
end
if nargin<2 || isempty(d)
    d=1;
end
typ=upper(typ);
s=s(:);

switch typ
    case {'GOE','O'} % beta==1
         p=(1/d)*((s/d)*pi/2).*exp(-(pi/4)*(s/d).^2);
    case {'GUE','U'} % beta==2
         p=(1/d)*((32/pi^2)*(s/d).^2).*exp(-(4/pi)*(s/d).^2);
    case {'POISSON','P'}
         p=(1/d)*exp(-(s/d));
end

ds=[diff(s);s(end)-s(end-1)];
nrm=trapz(p.*ds);

if nargout==0
    figure;
    plot(s,p);
    grid on;
    title('level spacing distribution');
    xlabel('s');
end
