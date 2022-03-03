% function mk_band_nnz_figs;
% generate png figs describing the removal of links; make avi video clip. 
% In: removedEdgelinks[1,ndst] a list of removed links
% Out: png-figs in '/BAND/nnz'
%      avi clip in 'BAND'

v = VideoWriter([dir_fig,'/','nnz.avi']);
open(v);
figure;
axis ij;
axis tight;
set(gca,'nextplot','replacechildren'); 
Fmean=mean(F,3);
ndst=size(Fmean,1)-1;
for n=1:10:ndst
    B=mk_banded(n,N);%imshow(B,[]);colorbar;
    img=Fmean.*B;
    img(img>0)=1;
    imagesc(img);colorbar;
    title(['n=', num2str(n,'%03u'),'   nnz=', num2str(nnz(img),'%05u')]);
    frm = getframe(gcf);
    writeVideo(v,frm);
    saveas(gcf,[dir_fig,'/','nnz','/',num2str(n,'%05u'),'.png']);
    fprintf('%u\t',n);
    if mod(n,50)==0;fprintf('\n'); end
end
close(v);
