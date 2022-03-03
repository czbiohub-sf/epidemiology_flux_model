% plot cent_toll
% In: res_cent.mat
% Out:summary3.mat

clear; 
close all;
plotFig6=0;
plotFig7=1;
plotFig8=1;
plotFig9=0;
plotFig10=1;
plotFig11=0;
tic;load('res_cent.mat');toc;

KL0=(log(pi)-double(eulergamma))/2; %  WD || poisson
ndst=length(D);
N=1024;
bb=N^2-Nnzero(1:ndst);
rng=[1:150,160:10:ndst];
% D=D(1:ndst);
% D(D<0)=eps;

if plotFig6
    f6=figure; %6
    plot(bb(rng),KL(2,:)/KL0,'-'); grid on;
    xlabel('iteration (# cuts)')
    ylabel('KL/KL_{0}');
    title(['relative KL divergence vs. # cuts,',...
        ' KL_{0}=',num2str(KL0,'%2.4f'), ' is D(WD||Poisson)']);
end

H=1-KL(2,:); %H=1-D; %entropy
Hs=smooth(H,11,'sgolay',3);
Hs=smooth(H,11);
%Hs=H; % unsmoothed
if plotFig7
   f7=figure;
   plot(bb(rng),Hs,'b.-'); grid on;
   ylabel('S');
   xlabel('c (# cuts)');
   xlim([0.5 ,5.5]*1e5);
   %title('Entropy Vs. num-cuts');
  end
%saveas(gcf,'f1','png');
%saveas(gcf,'f1','svg');

omega=mean(Tns,2);
omegaR=omega/omega(1);
err=std(Tns,0,2)/2; % half-size for plus/minus
errR=err/omega(1);

if plotFig8
   omegaR1=omegaR(rng);
   errR1=errR(rng);
   bb1=bb(rng); 
   Ix=[1:5:125,130:10:220,235:55:435];
   omegaR1=omegaR1(Ix); 
   errR1=errR1(Ix); 
   bb1=bb1(Ix); 
   f8=figure;
   errorbar(bb1,omegaR1,errR1,'b.-'); grid on; 
   ylabel('\Omega/\Omega_0');
   xlabel('c (# cuts)');
   %xlim([0.5 ,5.5]*1e5);
   %title('Epidemic size Vs. num-cuts');
end
%saveas(gcf,'f2','png');
%saveas(gcf,'f2','svg');

NnzeroR=Nnzero/Nnzero(1);
if plotFig9
   f9=figure;
   plot(bb(rng),Hs,'.-',bb(rng), omegaR(rng),'.-',...
       bb(rng), NnzeroR(rng),'.-'); grid on;
   legend('S','\Omega/\Omega_0','nnz');
   xlabel('c (# cuts)');
   %title('Entropy and Epidemic size Vs. num-cuts');
end
if plotFig10
   f10=figure;
   yyaxis left;
   pt1=plot(bb(rng),Hs,'-');
   yyaxis right;
   rAx=gca;
   pt2=plot(bb(rng), omegaR(rng),'-'); 
   grid on;
   legend([pt1,pt2],'S','\Omega/\Omega_0',...
       'Location','NorthWest');
   xlabel('c (# cuts)');
   xlim([0.5 ,5.4]*1e5);
   %title('Entropy and Epidemic size Vs. num-cuts');
end
%saveas(gcf,'f3','png');
%saveas(gcf,'f3','svg');
if plotFig11
   f11=figure;
   plot(Hs,omegaR(rng),'.-'); grid on;
   ylabel('\Omega/\Omega_0');
   xlabel('S');
   %title('Epidemic size Vs. entropy');
end
Nc=zeros(ndst,1);

%load('rmv_band_toll.mat','Nnzero','removedEdgeList');
for n=1:ndst
    Nc(n)=length(removedEdgeList{n});
    disp([Nc(n),N*N-Nnzero(n)]);
end
Nc=cumsum(Nc); %CENT is different than DIST 

figure;
plot(Nc(rng)/N/(N-1),Hs,'.-','DisplayName','S'); grid on;
hold on;
plot(Nc(rng)/N/(N-1),omegaR(rng),'.-','DisplayName','\Omega/\Omega_0');
hold off;
legend('Location','SouthWest');
xlabel('N_c/N/(N-1)'); 
ylim([0.85,1]);

save('summary3.mat','Hs','omegaR','Nc','Nnzero','rng');

