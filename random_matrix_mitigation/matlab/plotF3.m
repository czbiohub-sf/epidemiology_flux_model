% display level-spacing
if plotFig3
    %f3=figure; %3
    bar(s1,ps1,'DisplayName','hist'); grid on;  hold on;
    plot(wdy1.sval,wdy1.pval,'DisplayName','GOE');
    plot(pois1.sval,pois1.pval,'DisplayName','Poisson');
    hold off;
    xlim([0,10*d1]);
    legend;
    xlabel('s');
    ylabel('p(s)');
    title(['raw level spacing: N=',num2str(N),', #samp=',num2str(nsamp)]);
end