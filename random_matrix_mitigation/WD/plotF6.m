% display  improved unfolded level spacing
if plotFig6
    %f6=figure; %6 
    bar(s1,p1,'DisplayName','safeGraph'); grid on; 
    hold on;
    plot(s1,q2,'DisplayName','WD(GOE)');
    plot(s1,q3,'DisplayName','Poisson');
    hold off;
    text(1.0,1.0,['KL(safeGraph||Poisson)=', num2str(KL2,'%2.4f')]);
    xlim([0,5]);
    ylim([0,1.1]);
    legend;
    xlabel('s');
    ylabel('p(s)');
     %title(['unfolded level spacing: N=',num2str(N),', #samp=',num2str(nsamp)]);
end