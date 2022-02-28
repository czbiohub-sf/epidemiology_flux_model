%display  DoS
if plotFig2
    %f2=figure; %2
    h=histogram(Q,10*nbins1,'normalization','pdf'); grid on;
    nrm=sum(h.Values)*h.BinWidth;
    xlim([-1,1]*1e-4);
    xlabel('E');
    ylabel('\rho');
    title(['orig DoS: N=',num2str(N),', #samp=',num2str(nsamp),...
        ' (norm=',num2str(nrm,'%2.4f'),')']);
end