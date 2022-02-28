% function mk_band_nnz_figs;
% generate png figs describing the removal of links; make avi video clip. 
% In: removedEdgeLink[1,ndst] a list of removed links
% Out: png-figs in '/CENT/nnz'
%      avi clip in 'CENT'

if ~exist('removedEdgeList','var')
    load([dir_fig,'/','rmv_cent_toll.mat'],'removedEdgeList','ndst');
end



v = VideoWriter([dir_fig,'/','nnz.avi']);
open(v);
figure;
axis ij;
axis tight;
set(gca,'nextplot','replacechildren'); 
img=mean(F,3);
for n=1:ndst
    Ix=removedEdgeList{n};
    img(Ix)=0;
    img(img~=0)=1;
    imagesc(img);colorbar;
    title(['n=', num2str(n,'%04u'),'   nnz=', num2str(nnz(img),'%06u')]);
    frm = getframe(gcf);
    writeVideo(v,frm);
    saveas(gcf,[dir_fig,'/','nnz','/',num2str(n,'%05u'),'.png']);
end
close(v);
