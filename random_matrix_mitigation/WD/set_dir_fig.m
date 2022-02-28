%function set_dir_fig(dir_fig)
% []=set_dir_fig makes sub-directories for figures
% In:  dir_fig directory for generated figures
% Out: 
% Note: nnz sub-dir for figures showing the non-zero elements of the flux matrix
%      dos1 density-of-states (raw)
%      dos2 density-of-states (unfolded, i.e. uniform)
%      raw  level spacing distribution (raw)
%      unf  level spacing distribution (unfolded)
%      eval eigenvalues
 
if ~exist([dir_fig,'/','nnz'],'dir')
    mkdir([dir_fig,'/','nnz']);
end 
if ~exist([dir_fig,'/','dos1'],'dir')
    mkdir([dir_fig,'/','dos1']);
end
if ~exist([dir_fig,'/','dos2'],'dir')
    mkdir([dir_fig,'/','dos2']);
end
if ~exist([dir_fig,'/','raw'],'dir')
    mkdir([dir_fig,'/','raw']);
end
if ~exist([dir_fig,'/','unf'],'dir')
    mkdir([dir_fig,'/','unf']);
end
if ~exist([dir_fig,'/','eval'],'dir')
        mkdir([dir_fig,'/','eval']);
end

