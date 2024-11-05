close all
clear
clc

addpath(genpath("../../sparse-grids-matlab-kit"))

%% create sparse grid and save to file
clear

% \theta_s, \theta_r, \alpha, m, K_s

N = 5;
knots = {@(n)knots_CC(n,.3,.6), @(n)knots_CC(n,0,.08), @(n)knots_CC(n,.1,2), @(n)knots_CC(n,1.2,3), @(n)knots_CC(n,-8,-4)}; 
lev2knots = @lev2knots_doubling;

w = 3;
S = create_sparse_grid(N,w,knots,lev2knots);
Sr = reduce_sparse_grid(S);

export_sparse_grid_to_file(Sr)


% %% plot sparse grid
% 
% for i=1:length(Sr)
%     plot_sparse_grid(Sr(i),[4,5])
%     pause
%     hold on
% end
