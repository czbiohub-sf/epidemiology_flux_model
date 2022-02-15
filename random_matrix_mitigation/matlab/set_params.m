% set_params
% Use: strain_list
%
% NOTE: change directory of flux matrices (line #20)

%% flags for plotting
plotFig2=1;
plotFig3=1;
plotFig4=1;
plotFig5=1;
plotFig6=1;

%% parameters
typDiag='zero'; % the diagonal already set to zero
isBanded=0;     % make a banded matrix
tolKL=0.1;
smoothKL=0;     % smooth ps before calculating KL
KL0=(log(pi)-double(eulergamma))/2; %  D[WD || Poisson]
tolScale=1e6;   % used in flux2tol
%% directory of flux matrices
switch computer
    case 'PCWIN64'
        dirName=['C:/Users/Reuven-Pnini/Desktop/DATA','/','symmetricFlux'];
    case 'MACI64'
        %dirName=['/Users/reuven-pnini/MATLABWD/rmtmac/Matlab_safegraph','/','symmetricFlux'];
        strpath=pwd;
        ix=strfind(strpath,'/WD');
        dirName=[strpath(1:ix-1),'/','symmetricFlux'];
        clear('strpath','ix');
end
flist=strain_list(dirName);
%% chk for the consistentcy of sizes:
ff=load([dirName,'/',flist{1}]);
[N1,N2]=size(ff.F);
[N3,N4]=size(ff.Md);
if N1~=N2 || N1~=N3
    error('square matrix expected');
end
if N4~=1
    error('a column vector is expected');
end
N=N1; clear('N1','N2','N3','N4');
clear('ff');
