% display unfolded level spacing
if plotFig5
    %f5=figure; %5
    %hold on;
    bar(s2,ps2,'DisplayName','hist'); grid on; hold on;
    plot(wdy2.sval,wdy2.pval,'DisplayName','GOE');
    plot(pois2.sval,pois2.pval,'DisplayName','Poisson');
    hold off;
    xlim([0,5*d2]);
    text(1,0.9,['\langle{s}\rangle=',num2str(d2)]);
    text(1,0.8,['KL(safeGraph||Poisson)=', num2str(KL1,'%2.4f')]);
    legend;
    xlabel('s');
    ylabel('p(s)');
    title(['unfolded level spacing: N=',num2str(N),', #samp=',num2str(nsamp)]);
end