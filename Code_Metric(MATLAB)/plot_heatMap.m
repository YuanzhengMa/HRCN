clc
%clear all
cd 
path = ' C:\Users\Administrator\Desktop\processed\plot_heatmap_real\row3\';%' C:\Users\Administrator\Desktop\need\fig6\row2\'%
eval(strcat('cd',path));
D = dir(strcat(path(2:end), '*.jpg'));
C = {};
for idx = 1:numel(D)
    C(idx) = cellstr(D(idx).name);
end
C = natsort(C);

f1 = figure(1);
axes1 = axes('Parent',f1);
legend_name = {};
MEAN = [];
VAR = [];
MAX=[];
MIN=[];
for idx  = 1:numel(C)
    name_col =  split(C(idx),'_');
    name = char(name_col(3));
    eval(strcat('temp = imread(char(C(idx)));'));
    eval(strcat(name,'=rgb2gray(temp);'));
    %if strcmp(name(1:3), 'Inp')
       eval(strcat(name,' = imresize(', name,',size(GroundTrue),''nearest'');'));
    %end
    eval(strcat('cmp_obj =',name,';'));
    

%     if strcmp(path(end-1),'1')
%         eval(strcat(name,'_crop =',name,'(131:210, 153:232);'));%1
%         cmp_obj = 'Bicubic_crop';
%     elseif strcmp(path(end-1),'2')
%         %eval(strcat(name,'_crop =',name,'(106:185, 251:330);'));%2
%         eval(strcat(name,'_crop =',name,'(201:280, 286:365);'));%2
%         cmp_obj = 'SRUNet_crop';
%     elseif strcmp(path(end-1),'3')
%         eval(strcat(name,'_crop =',name,'(246:325, 186:265);'));%3
%         cmp_obj = 'DBPN_crop';
%     end
end

for idx = 1:numel(C)
    name_col =  split(C(idx),'_');
    name = char(name_col(3));
    eval(strcat(name,"_delta = double(GroundTrue)-double(",name,");"));
    %需要截图时候用这个
    %eval(strcat(name,"_delta = double(",name,");"));
    row_num = split(path,'\');
    row_num = char(row_num(end-1));
    if strcmp(row_num,'row1')
        eval(strcat(name,"_crop = ",name,"(151:230,286:365);"));%(151:230,286:365)非GT(126:205,281:360)
    elseif strcmp(row_num,'row2')
        eval(strcat(name,"_crop = ",name,"(268:347,116:195);"));%如果GT(268:347,116:195),非(261:340,121:200)%(176:255,176:255)
    else
        eval(strcat(name,"_crop = ",name,"(126:205,301:380);"));%(126:205,281:360)
    end
    eval(strcat("imagesc(",name,"_delta);"));
    eval(strcat("imwrite(double(",name,"_delta)/255,","'img_x_",name,"_fortomo.jpg');"));
    eval(strcat("temp = ",name,"_delta ;"));
    eval(strcat("save ","img_x_",name,"_fortomo.mat ","temp ;"));
    eval(strcat(name,'_max =max(max(',name,'_delta));'));
    eval(strcat('MAX=horzcat(MAX,',name,'_max);'));
    
    eval(strcat(name,'_min =min(min(',name,'_delta));'));
    eval(strcat('MIN=horzcat(MIN,',name,'_min);'));
    
    eval(strcat(name,'_mean =mean(mean(abs(',name,'_delta)));'));
    eval(strcat('MEAN=horzcat(MEAN,',name,'_mean);'));
    
    eval(strcat(name,'_var =std(',name,'_delta,0,''all'');'));
    eval(strcat('VAR=horzcat(VAR,',name,'_var);'));
    figure(1)
    %需要delta时候用这个
    colormap(white);
    %需要截图时候用这个
    %colormap(gray);
    caxis([-100, 100]);
    % set(axes1,'FontName','Times');
    % colorbar;
    % set(gca, 'FontSize', 20);
    saveas(gca, strcat('Diff btw target and ',name,'.eps'), 'psc');
    %需要delta时候用这个
    colormap(white);
    %需要截图时候用这个
    %colormap(gray);
    saveas(gca, strcat('Diff btw target and ',name,'.png'));
    saveas(gca, strcat('Diff btw target and ',name,'.fig'));
    figure(2)
    colormap(gray);
    caxis([0, 255]);
    
    eval(strcat("imagesc(",name,"_crop);"));
    saveas(gca, strcat('Diff btw target and ',name,'_crop.eps'), 'psc');
    saveas(gca, strcat('Diff btw target and ',name,'_crop.png'));
end
% eval(strcat("imagesc(double(GroundTrue)-double(",cmp_obj,"));"));
% colormap(white);
% caxis([-127, 128]);
% % set(axes1,'FontName','Times');
% % colorbar;
% % set(gca, 'FontSize', 20);
% saveas(gca, strcat('Diff btw target and bicubic.eps'), 'psc');
% saveas(gca, strcat('Diff btw target and bicubic.fig'));
% figure,
% imagesc(double(GroundTrue_crop)-double(Res3Net_crop));
% colormap(white);
% caxis([-127, 128]);
% % set(axes1,'FontName','Times');
% % colorbar;
% % set(gca, 'FontSize', 20);
% saveas(gca, strcat('Diff btw target and forged.eps'), 'psc');
% saveas(gca, strcat('Diff btw target and forged.fig'));
UP = [GroundTrue_max,Input_max,SRResNet_max,DBPN_max,SRUNet_max,ASR_max];
DOWN = [GroundTrue_min,Input_min,SRResNet_min,DBPN_min,SRUNet_min,ASR_min];
MID =  [GroundTrue_mean,Input_mean,SRResNet_mean,DBPN_mean,SRUNet_mean,ASR_mean];
VAR =  [GroundTrue_var,Input_var,SRResNet_var,DBPN_var,SRUNet_var,ASR_var];
legend_name = {'GroundTrue','Input','SRResNet','DBPN','SRUNet','ASR'}
% UP = [GroundTrue_max,Input_max,Bicubic_max,ASR_max,ASRCOCO_max,ASRCOCOPlus_max];
% MID = [GroundTrue_mean,Input_mean,Bicubic_mean,ASR_mean,ASRCOCO_mean,ASRCOCOPlus_mean];
% VAR = [GroundTrue_var,Input_var,Bicubic_var,ASR_var,ASRCOCO_var,ASRCOCOPlus_var];
% legend_name = {'GroundTrue','Input','Bicubic','ASR','ASRCOCO','ASRCOCOPlus'}
MARKER = ['ro';'g+';'b.';'c*';'mx';'k^']
%%
f1 = figure(100);
axes1 = axes('Parent',f1);
for x = 1:numel(UP)
   plot(MID(x),VAR(x),MARKER(x,:),'LineWidth',2,'MarkerSize' ,10);
   hold on
end
grid on

legend(legend_name,'NumColumns',6)
set(gca, 'FontSize', 20);
legend1 = legend(axes1,'show');
set(legend1,'FontSize',18);
ylim([0,20]);
xlim([0,10]);
%set(gca,'xticklabel',[]); 
%set(gca,'yticklabel',[]);
% xlabel('Max difference');
% ylabel('Min difference');
%legend(['LR','SRResNet','DBPN','SRUNet','ASR']);
%% search for max and its loc
eval(strcat("temp1=double(target_crop)-double(",cmp_obj,");"));
temp2 = double(target_crop)-double(forged_crop);
M1 = 0;
M2 = 0;
m1 = 0;
m2 = 0;
for idx = 1:size(temp1,1)
    for idy = 1:size(temp1,2)
        if temp1(idx, idy) > M1
            M1_idx = idx;
            M1_idy = idy;
            M1 = temp1(idx,idy);
        end
        if temp1(idx, idy) < m1
            m1_idx = idx;
            m1_idy = idy;
            m1 = temp1(idx,idy);
        end
        % 
        if temp2(idx, idy) > M2
            M2_idx = idx;
            M2_idy = idy;
            M2 = temp2(idx,idy);
        end
        if temp2(idx, idy) < m2
            m2_idx = idx;
            m2_idy = idy;
            m2 = temp2(idx,idy);
        end
    end
end