clear all
close all
clc

% name = 'Ambrozek';
name = 'Bednarova';
% name = 'Cip';

path_data = ['C:\Data\Jakubicek\MRI_Brain\' name '\Outputs\comp\'];


path_data_orig = [path_data '\DCE_cut.nii.gz'];
path_data_reg = [path_data '\DCE_cut_reg3.nii.gz'];
% path_data_orig = [path_data '\DCE_orig.nii.gz'];
% path_data_reg = [path_data '\DCE_orig_reg2.nii.gz'];

path_save = 'D:\Projekty\BrainFNUSA\MRI_gliom\Export';
mkdir(path_save)

%% 

fix = niftiread(path_data_orig);
reg = niftiread(path_data_reg);

fix = fix(:,:,:,2:end);

std_fix = std(single(fix),1,4);
std_reg = std(single(reg),1,4);

mean(std_fix,"all")
mean(std_reg,"all")

%% MIP of std

MIP_fix_AX = max(std_fix,[],3);
MIP_reg_AX = max(std_reg,[],3);

MIP_fix_SAG = squeeze(max(std_fix,[],2));
MIP_reg_SAG = squeeze(max(std_reg,[],2));

MIP_fix_COR = squeeze(max(std_fix,[],1));
MIP_reg_COR = squeeze(max(std_reg,[],1));

range = [0,800];
N = 512;
figure(1)
% subplot 231
imagesc(MIP_fix_AX,range)
colormap(viridis(N))
axis off
% saveas(gcf,'Barchart.png')
figure
% subplot 234
imagesc(MIP_reg_AX,range)
colormap(viridis(N))
axis off
figure
% subplot 232
imagesc(MIP_fix_SAG,range)
colormap(viridis(N))
axis off
figure
% subplot 235
imagesc(MIP_reg_SAG,range)
colormap(viridis(N))
axis off
figure
% subplot 233
imagesc(MIP_fix_COR,range)
colormap(viridis(N))
axis off
figure
% subplot 236
imagesc(MIP_reg_COR,range)
colormap(viridis(N))
axis off

%% analzsis std max

% C_fix = squeeze(max(std_fix,[],[1,2]));
% C_reg = squeeze(max(std_reg,[],[1,2]));
C_fix = squeeze(mean(std_fix,[1,2]));
C_reg = squeeze(mean(std_reg,[1,2]));

figure(2)
plot(C_fix)
hold on
plot(C_reg)

% C_fix = squeeze(max(std_fix,[],[1,2]));
% C_reg = squeeze(max(std_reg,[],[1,2]));
C_fix = squeeze(mean(std_fix,[3,1]));
C_reg = squeeze(mean(std_reg,[3,1]));

figure(3)
plot(C_fix)
hold on
plot(C_reg)

% C_fix = squeeze(max(std_fix,[],[1,2]));
% C_reg = squeeze(max(std_reg,[],[1,2]));
C_fix = squeeze(mean(std_fix,[3,2]));
C_reg = squeeze(mean(std_reg,[3,2]));

figure(4)
plot(C_fix, LineWidth=2)
hold on
plot(C_reg, LineWidth=2)

%% For ortho slices

sl_ax = 42;
Orth_fix_AX = std_fix(:,:,sl_ax);
Orth_reg_AX = std_reg(:,:,sl_ax);

sl_sag = 42;
Orth_fix_SAG = squeeze(std_fix(:,sl_sag,:));
Orth_reg_SAG = squeeze(std_reg(:,sl_sag,:));

sl_cor = 48;
Orth_fix_COR = squeeze((std_fix(sl_cor,:,:)));
Orth_reg_COR = squeeze((std_reg(sl_cor,:,:)));

range = [0,300]; 
N = 512;

figure(20)
% subplot 231
imagesc(Orth_fix_AX,range)
colormap(viridis(N))
axis off
% saveas(gcf,[path_save '\Fix_AX.png'])
% export_fig([path_save '\Fix_AX.png'],'-png')
colorbar

figure(20)
% subplot 234
imagesc(Orth_reg_AX,range)
colormap(viridis(N))
axis off
% saveas(gcf,[path_save '\Reg_AX.png'])
export_fig([path_save '\Reg_AX.png'],'-png')

figure(20)
% subplot 232
imagesc(Orth_fix_SAG,range)
colormap(viridis(N))
axis off
% saveas(gcf,[path_save '\Fix_SAG.png'])
export_fig([path_save '\Fix_SAG.png'],'-png')

figure(20)
% subplot 235
imagesc(Orth_reg_SAG,range)
colormap(viridis(N))
axis off
% saveas(gcf,[path_save '\Reg_SAG.png'])
export_fig([path_save '\Reg_SAG.png'],'-png')

figure(20)
% subplot 233
imagesc(Orth_fix_COR,range)
colormap(viridis(N))
axis off
% saveas(gcf,[path_save '\Fix_COR.png'])
export_fig([path_save '\Fix_COR.png'],'-png')

figure(20)
% subplot 236
imagesc(Orth_reg_COR,range)
colormap(viridis(N))
axis off
% saveas(gcf,[path_save '\Reg_COR.png'])
export_fig([path_save '\Reg_COR.png'],'-png')

mean(cat(1,Orth_fix_SAG(:),Orth_fix_COR(:),Orth_fix_AX(:)),"all")
mean(cat(1,Orth_reg_SAG(:),Orth_reg_COR(:),Orth_reg_AX(:)),"all")



%%
close all
% figure(25)
% plot(Orth_fix_AX(:,43),Color=[0.8500, 0.3250, 0.0980], LineWidth=2)
% hold on
% plot(Orth_reg_AX(:,43),Color=[0, 0.4470, 0.7410], LineWidth=2)
% figure(25)
% plot(Orth_fix_COR(:,43),Color=[0.8500, 0.3250, 0.0980], LineWidth=2)
% hold on
% plot(Orth_reg_COR(:,43),Color=[0, 0.4470, 0.7410], LineWidth=2)

figure(25)
plot(Orth_fix_SAG(:,43),Color=[0.8500, 0.3250, 0.0980], LineWidth=2)
hold on
plot(Orth_reg_SAG(:,43),Color=[0, 0.4470, 0.7410], LineWidth=2)

% xlabel('space coordinate','FontSize',28)
% ylabel('standard deviation','FontSize',28)
xticks([0,35,70])
yticks([0,260,520])

g = get(gcf,'Position');
g(3)=1000;
set(gcf,"Position",g)
set(gca,'FontSize',28)
legend({'Original','Registered'})
% export_fig([path_save '\curve_brain_AX.png'],'-png')
box off
ax = gca;
ax.LineWidth = 3;
xlim([0,70])
ylim([0,520])

% export_fig([path_save '\curve_brain_AX.png'],'-png')
print([path_save '\curve_brain_SAG.png'],'-dpng')


%% example of results
close all
sl_ax = 42;
time_1 = 48;
time_2 = 35;
% fix_AX_1 = fix(:,:,sl_ax,time_1);
% fix_AX_2 = fix(:,:,sl_ax,time_2);
% reg_AX_1 = reg(:,:,sl_ax,time_1);
% reg_AX_2 = reg(:,:,sl_ax,time_2);
fix_AX_1 = fix(:,:,100,40);
fix_AX_2 = fix(:,:,100,30);
reg_AX_1 = reg(:,:,51,20);
reg_AX_2 = reg(:,:,51,15);

% figure(20)
% subplot 121
% imshowpair(fix_AX_1,fix_AX_2)
% subplot 122
% imshowpair(reg_AX_1,reg_AX_2)

figure(20)
imshowpair(fix_AX_1,fix_AX_2)
g = get(gcf,'Position');
% g(1:2)=0;
g(3:4)=g(3:4)*10;
set(gcf,"Position",g)
% print([path_save '\beforeBR.png'],'-dpng')
export_fig([path_save '\beforeBR.png'],'-png')

figure(20)
imshowpair(reg_AX_1,reg_AX_2)
g = get(gcf,'Position');
% g(1:2)=0;
g(3:4)=g(3:4)*10;
set(gcf,"Position",g)
% print([path_save '\afterBR.png'],'-dpng')
export_fig([path_save '\afterBR.png'],'-png')


