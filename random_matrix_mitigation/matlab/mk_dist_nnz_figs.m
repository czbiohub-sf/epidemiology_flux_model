% function mk_dist_nnz_figs;
% generate png figs describing the removal of links; make avi video clip. 
% In: removedEdgeList[1,ndst] a list of removed links
% Out: png-figs in '/DIST/nnz'
%      avi clip in 'DIST'
if ~exist('removedEdgeList','var')
    load([dir_fig,'/','rmv_dist_toll.mat'],'removedEdgeList');
end

v = VideoWriter([dir_fig,'/','nnz.avi']);
open(v);
figure;
axis ij;
axis tight;
set(gca,'nextplot','replacechildren'); 
Fmean=mean(F,3);
for n=1:length(removedEdgeList)
    Ix=removedEdgeList{n};
    img=Fmean;
    img(Ix)=0;
    img(img~=0)=1;
    imagesc(img);colorbar;
    title(['n=', num2str(n,'%03u'),'   nnz=', num2str(nnz(img),'%05u')]);
    frm = getframe(gcf);
    writeVideo(v,frm);
    saveas(gcf,[dir_fig,'/','nnz','/',num2str(n,'%05u'),'.png']);
end
close(v);
