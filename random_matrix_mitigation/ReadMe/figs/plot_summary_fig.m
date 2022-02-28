% fig4
clear;
close all;
s1=load('summary1.mat');
s2=load('summary2.mat');
s3=load('summary3.mat');

%% fig 16
if 0
ind=[1:40,56:93];
s3.Hs=s3.Hs(ind);
s3.omegaR=s3.omegaR(ind);
s2.omegaR=s2.omegaR(s2.rng);
omegaS=smooth(s3.omegaR,15,'sgolay',3);
figure;
plot(s1.Hs,s1.omegaR,'.-','DisplayName','DIST'); grid on;
hold on;
plot(s2.Hs,s2.omegaR,'.-','DisplayName','BAND');
plot(s3.Hs,omegaS,'.-','DisplayName','CENT');
hold off;
legend;
xlabel('S');
ylabel('\Omega/\Omega_0');
xlim([0.875, 0.95]);
saveas(gcf,'f16','png');
saveas(gcf,'f16','svg');
end

if 1
N=1024;    
%DIST (58)    
Hs1=s1.Hs;
omegaR1=s1.omegaR;
Nc1=s1.Nc;
%BAND (1023)
Hs2=s2.Hs;
omegaR2=s2.omegaR;
Nc2=s2.Nc;
%CENT (435)
Hs3=s3.Hs;
omegaR3=s3.omegaR(s3.rng);
Nc3=s3.Nc(s3.rng);
%% fig17
figure;
plot(Nc1/N/(N-1),Hs1,'.-','DisplayName','distance cutoff'); 
grid on;
hold on;
plot(Nc2/N/(N-1),Hs2,'-','DisplayName','band cutoff'); 
plot(Nc3/N/(N-1),Hs3,'-','DisplayName','edge-centrality'); 
hold off;
legend('Location','NorthWest');
xlabel('Nc/N/(N-1)');
ylabel('S');
%saveas(gcf,'f17','png');
saveas(gcf,'f17','svg');
savefig(gcf,'f17','compact');
%% fig18
figure;
plot(Nc1/N/(N-1),omegaR1,'.-','DisplayName','distance cutoff');
grid on;
hold on;
plot(Nc2/N/(N-1),omegaR2,'-','DisplayName','Band cutoff'); 
plot(Nc3/N/(N-1),omegaR3,'-','DisplayName','edge-centrality'); 
hold off;
legend('Location','SouthWest');
xlabel('Nc/N/(N-1)');
ylabel('\Omega/\Omega_0');
ylim([0.5,1]);
%saveas(gcf,'f18','png');
saveas(gcf,'f18','svg');
savefig(gcf,'f18','compact');

end