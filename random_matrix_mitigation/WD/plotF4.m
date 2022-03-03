%display flat DoS
if plotFig4
    %f4=figure; %4
    h=histogram(G(:),nbins2,'normalization','pdf'); grid on;
    nrm=sum(h.Values)*h.BinWidth;
    xlim([0,N+26]);%1024+26=1050
    xlabel('\epsilon');
    ylabel('\rho');
    title(['Unfolded DoS: N=',num2str(N),', #samp=',num2str(nsamp),...
        ' (norm=',num2str(nrm,'%2.4f'),')']);
end