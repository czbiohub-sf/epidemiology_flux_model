% plot dist_toll
% In: res_dist.mat
% Out:summary1.mat

clear; 
close all;

plotFig6=0;
plotFig7=1;
plotFig8=1;
plotFig9=0;
plotFig10=1;
plotFig11=0;
tic;load('res_dist.mat');toc;
KL0=(log(pi)-double(eulergamma))/2; %  WD || poisson
ndst=length(dd);

if plotFig6
    f6=figure; %6
    plot(1:ndst,KL(2,:)/KL0,'.-'); grid on;
    xlabel('iteration (# cuts)')
    ylabel('KL/KL_{0}');
    title(['relative KL divergence vs. # cuts,',...
        ' KL_{0}=',num2str(KL0,'%2.4f'), ' is D(WD||Poisson)']);
end
H=1-KL(2,:); %1-D; %entropy
%Hs=smooth(H,25,'sgolay',3);
Hs=smooth(H,5);
if plotFig7
   f7=figure;
   plot(dd/1000,Hs,'b.-'); hold on;
   plot([0,dd(1)/1000],[1,Hs(1)],'b:'); grid on; hold off;
   %plot([0,dd/1000],[1;Hs],'.-'); grid on; hold off;
   ylabel('S');
   xlabel('d_c [km]');
   %title('Entropy Vs. distance cut-off');
   ylim([0.87, 0.97]);
end
% saveas(gcf,'f1','png');
% saveas(gcf,'f1','svg');

omega=mean(Tns,2);
omegaR=omega/omega(end);
err=std(Tns,0,2)/2; % half-size for plus/minus
errR=err/omega(end);
if plotFig8
   f8=figure;
   errorbar(dd/1000,omegaR,errR,'b.-'); grid on; 
   ylabel('\Omega/\Omega_0');
   xlabel('d_c [km]');
   %title('Epidemic size Vs. distance cut-off');
end
% saveas(gcf,'f2','png');
% saveas(gcf,'f2','svg');

NnzeroR=Nnzero/Nnzero(1);
if plotFig9
   f9=figure;
   plot(dd/1000,Hs,'.-',dd/1000, omegaR,'.-',...
       dd/1000, NnzeroR,'.-'); grid on;
   legend('S','\Omega/\Omega_0','nnz');
   xlabel('d_c [km]');
   %title('Entropy and Epidemic size Vs. distance cut-off');
end
if plotFig10
   f10=figure;
   yyaxis left;
   pt1=plot(dd/1000,Hs,'.-');
   Hax=gca;
   col=Hax.ColorOrder(1,:);
   yyaxis right;
   hold on;
   pt2=plot(dd/1000, omegaR,'.-'); 
   %plot([0,dd(1)/1000],[1,Hs(1)],'Color',col,'LineStyle','--');
   hold off; grid on;
   legend([pt1,pt2],'S','\Omega/\Omega_0',...
       'Location','NorthEast');
   xlabel('d_c [km]');
   ylim([0.74,1.01]);
   %title('Entropy and Epidemic size Vs. distance cut-off');
end
% saveas(gcf,'f3','png');
% saveas(gcf,'f3','svg');

if plotFig11
   f11=figure;
   plot(Hs,omegaR,'.-'); grid on;
   ylabel('\Omega/\Omega_0');
   xlabel('S');
   %title('Epidemic size Vs. entropy');
end

Nc=zeros(ndst,1);
N=1024;
for n=1:ndst
    Nc(n)=length(removedEdgeList{n});
end
figure;
plot(Nc/N/(N-1),Hs,'.-','DisplayName','S'); grid on;
hold on;
plot(Nc/N/(N-1),omegaR,'.-','DisplayName','\Omega/\Omega_0');
hold off;
legend('Location','SouthWest');
xlabel('N_c/N/(N-1)'); 

save('summary1.mat','Hs','omegaR','Nc')