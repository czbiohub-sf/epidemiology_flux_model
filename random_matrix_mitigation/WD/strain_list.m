function flist2=strain_list(fdir)
%In:	fdir  directory of files
%Out:   flist a list of file names

% if nargin==0
%    fdir=[pwd,'/','symmetricFlux'];
% end
d=dir(fdir);
flist1={d.name};
N=length(flist1)-2;
flist2=cell(1,N);
for n=1:N
    flist2(n)=flist1(n+2);
end
