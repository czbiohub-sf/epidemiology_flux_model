% plot band_toll
% In: res_band.mat
% Out:summary2.mat

clear; 
close all;
plotFig6=0;
plotFig7=1;
plotFig8=1;
plotFig9=0;
plotFig10=1;
plotFig11=0;
tic;load('res_band.mat');toc;
KL0=(log(pi)-double(eulergamma))/2; %  WD || poisson
ndst=length(D);
bb=1:ndst;
D(D<0)=eps;
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
Hs=smooth(H,121);
if plotFig7
   f7=figure;
   plot(bb,Hs,'b-'); grid on;
   ylabel('S');
   xlabel('B (half-bandwidth)');
   xlim([0, 1024]);
   %title('Entropy Vs. distance cut-off');
  end
% saveas(gcf,'f1','png');
% saveas(gcf,'f1','svg');

omega=mean(Tns,2);
omegaR=omega/omega(end);
err=std(Tns,0,2)/2; % half-size for plus/minus
errR=err/omega(end);
if plotFig8
   f8=figure;
   errorbar(bb(1:20:end),omegaR(1:20:end),...
       errR(1:20:end),'b-'); grid on; 
   ylabel('\Omega/\Omega_0');
   xlabel('B (half-bandwidth)');
   xlim([0, 1024]);
   %title('Epidemic size Vs. band cut-off');
end
% saveas(gcf,'f2','png');
% saveas(gcf,'f2','svg');

NnzeroR=Nnzero/Nnzero(1);
if plotFig9
   f9=figure;
   plot(bb,Hs,'.-',bb, omegaR,'.-',...
       bb, NnzeroR,'.-'); grid on;
   legend('S','\Omega/\Omega_0','nnz');
   xlabel('B (band-width)');
   %title('Entropy and Epidemic size Vs. band cut-off');
end
if plotFig10
   f10=figure;
   yyaxis left;
   pt1=plot(bb,Hs,'-');
   yyaxis right;
   pt2=plot(bb, omegaR,'-'); 
   grid on;
   legend([pt1,pt2],'S','\Omega/\Omega_0',...
       'Location','NorthEast');
   xlabel('B (half-bandwidth)');
   xlim([0, 1024]);
   %title('Entropy and Epidemic size Vs. band cut-off');
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
load('rmv_band_toll.mat','Nnzero','removedEdgeList');
for n=1:ndst
    Nc(n)=length(removedEdgeList{n});
    disp([Nc(n),N*N-Nnzero(n)]);
end
figure;
plot(Nc/N/(N-1),Hs,'.-','DisplayName','S'); grid on;
hold on;
plot(Nc/N/(N-1),omegaR,'.-','DisplayName','\Omega/\Omega_0');
hold off;
legend('Location','SouthWest');
xlabel('N_c/N/(N-1)'); 

save('summary2.mat','Hs','omegaR','Nc');

