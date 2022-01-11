clear all
clc

%p = '  C:\Users\Administrator\Desktop\ENLARGED\FORPLOT\REALPLOTBOX_DURR3x2\'
%p = '  C:\Users\Administrator\Desktop\ENLARGED\FORPLOT\REALPLOTBOX_DURR3x4';
p = '  C:\Users\Administrator\Desktop\ENLARGED\FORPLOT\FIG_ROW1\';
D = dir(p(1:end));
eval(strcat('cd', p));

idy  = 1;
path = {};

for  idx = 3: numel(D)
     path(idx-2) = cellstr(strcat(D(idx).name,'\sub\'));
end
 
% read image
for idx = 1:length(path)
    Dir_temp = dir(char(strcat(path(idx),'*.jpg')));
    sum_L = length(Dir_temp);

%     for num = 1:sum_L
%         Ary_temp(num) = string(Dir_temp(num).name);
%     end

    for idy = 1:sum_L
        name = split(path(idx),'\');
        %if strcmp(char(name(5)),'best_checkpoints')
        prefix = char(name(1));
        im_num = split(Dir_temp(idy).name,'_');
        %%%%%%%%%%%%name = strcat(prefix,'_',char(im_num(2)));
        name = strcat(prefix,'_',num2str(idy));
        eval('temp = imread(char(strcat(path(idx),Dir_temp(idy).name)));');
        if size(temp,3) ~= 1
            eval([name,' = im2gray(temp);']);
        else
            eval([name,' = temp;']);
        end
        %eval(strcat('imwrite(temp,',strcat(path(idx),'_',name),');'));
        
    end
end
% metric
ary = {};
for idx = 1:numel(path)
    temp = split(path(idx),'\');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ary(idx) = temp(end-2);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
% init
for idx = 1:numel(ary)
    eval(strcat('sum_MSE_', char(ary(idx)),' =0;'));
    eval(strcat('sum_PSNR_', char(ary(idx)),' =0;'));
    eval(strcat('sum_MSSIM_', char(ary(idx)),' =0;'));
end

for idx = 1:sum_L
    disp(strcat("img",string(idx)," ready to comp!"));
    % read raw image
    name = split(path(1),'\');
    name = strcat('GroundTrue_',num2str(idx));
    eval(strcat('raw =double(', name,');'));
    % read AA-EE image
    idz = 1;
    for idy = 1:numel(ary)
        disp(strcat("compare with ",ary(idy)));
        name = split(path(idy),'\');
        name = strcat(char(name(1)),'_',num2str(idx));
        if strcmp(name(1:3), 'Inp')
            eval(strcat(name,' = imresize(', name,',size(raw),''nearest'');'));
        end
%         if strcmp(name(1:3), 'Bic')
%             eval(strcat(name,' = imresize(', name,',size(raw),''bicubic'');'));
%         end
        eval(strcat('temp =double(', name,');'));
        % MSE
        eval(strcat('MSE_',char(ary(idy)),'=squeeze(sum(sum((raw-temp(1:size(raw,1),1:size(raw,2))).^2))./(400*400));'));
        eval(strcat('sum_MSE_', char(ary(idy)),'(idx)= MSE_',char(ary(idy)),';'))
        % MSSIM
        eval(strcat('MSSIM_',char(ary(idy)),'=multissim(temp(1:size(raw,1),1:size(raw,2)), raw);'));
        eval(strcat('sum_MSSIM_', char(ary(idy)), '(idx)=MSSIM_',char(ary(idy)),';'))
        % PSNR
        eval(strcat('PSNR_',char(ary(idy)),'=psnr(temp(1:size(raw,1),1:size(raw,2))/255, raw/255);'));
        eval(strcat('sum_PSNR_', char(ary(idy)),'(idx)=PSNR_',char(ary(idy)),';'))
        idz = idz + 1;
        figure(1),
        imshowpair(raw, temp,'montage');
        if idx == sum_L
            eval(strcat('avg_MSE_',char(ary(idy)),'=mean(sum_MSE_',char(ary(idy)),');'));
            eval(strcat('avg_PSNR_',char(ary(idy)),'=mean(sum_PSNR_',char(ary(idy)),');'));
            eval(strcat('avg_MSSIM_',char(ary(idy)),'=mean(sum_MSSIM_',char(ary(idy)),');'));
            %  append
            eval(strcat('MSE_bar_y(idy) =', 'avg_MSE_',char(ary(idy)),';'));
            eval(strcat('MSSIM_bar_y(idy) =', 'avg_MSSIM_',char(ary(idy)),';'));
            eval(strcat('PSNR_bar_y(idy) =', 'avg_PSNR_',char(ary(idy)),';'));
            % error  high
           eval(strcat('max_MSE_',char(ary(idy)),'=mean(((sum_MSE_',char(ary(idy)),'-avg_MSE_',char(ary(idy)),')>0).*(sum_MSE_',char(ary(idy)),'-avg_MSE_',char(ary(idy)),'));'));
            eval(strcat('max_PSNR_',char(ary(idy)),'=mean(((sum_PSNR_',char(ary(idy)),'-avg_PSNR_',char(ary(idy)),')>0).*(sum_PSNR_',char(ary(idy)),'-avg_PSNR_',char(ary(idy)),'));'));
            eval(strcat('max_MSSIM_',char(ary(idy)),'=mean(((sum_MSSIM_',char(ary(idy)),'-avg_MSSIM_',char(ary(idy)),')>0).*(sum_MSSIM_',char(ary(idy)),'-avg_MSSIM_',char(ary(idy)),'));'));
            % append
            eval(strcat('MSE_eh(idy,:) =', 'sum_MSE_',char(ary(idy)),';'));
            eval(strcat('MSSIM_eh(idy,:) =', 'sum_MSSIM_',char(ary(idy)),';'));
            eval(strcat('PSNR_eh(idy,:) =', 'sum_PSNR_',char(ary(idy)),';'));
            % error  low
            eval(strcat('min_MSE_',char(ary(idy)),'=mean(((sum_MSE_',char(ary(idy)),'-avg_MSE_',char(ary(idy)),')<0).*(sum_MSE_',char(ary(idy)),'-avg_MSE_',char(ary(idy)),'));'));
            eval(strcat('min_PSNR_',char(ary(idy)),'=mean(((sum_PSNR_',char(ary(idy)),'-avg_PSNR_',char(ary(idy)),')<0).*(sum_PSNR_',char(ary(idy)),'-avg_PSNR_',char(ary(idy)),'));'));
            eval(strcat('min_MSSIM_',char(ary(idy)),'=mean(((sum_MSSIM_',char(ary(idy)),'-avg_MSSIM_',char(ary(idy)),')<0).*(sum_MSSIM_',char(ary(idy)),'-avg_MSSIM_',char(ary(idy)),'));'));
            % append
            eval(strcat('MSE_el(idy,:) =', 'sum_MSE_',char(ary(idy)),';'));
            eval(strcat('MSSIM_el(idy,:) =', 'sum_MSSIM_',char(ary(idy)),';'));
            eval(strcat('PSNR_el(idy,:) =', 'sum_PSNR_',char(ary(idy)),';'));
        end
    end
end
%% SUM error bar
figure(1),
bar_x = repmat(ary,1,3*30);
% bar_x = {};
% for idx = 1:8
%     for idy = 1:36
%         bar_x((idx-1)*36+idy) = ary(idx);
%     end
% end
% bar_x = repmat(bar_x,1,3);
%color_sel = repmat({'x2','x3','x4'},1,size(MSE_eh, 2)*size(MSE_eh, 1)/3);
c0 = repmat({'MSE'},1,size(MSE_eh, 2)*size(MSE_eh, 1));
c1 = repmat({'PSNR'},1,size(PSNR_eh, 2)*size(PSNR_eh, 1));
c2 = repmat({'MSSIM'},1,size(MSSIM_eh, 2)*size(MSSIM_eh, 1));
color_sel = [c0,c1,c2];
temp=reshape([MSE_eh./max(MSE_eh,[],'all'),PSNR_eh./max(PSNR_eh(4:end,:),[],'all'),MSSIM_eh], 1, 3*size(MSE_eh, 1)*size(MSE_eh, 2));
temp_X = bar_x;
g=gramm('x', temp_X,'y',temp,'color', color_sel);%(MSE_bar_y);
g.geom_point('alpha',0.3,'dodge',0.1);
g.set_color_options('map','matlab');
g.set_text_options( 'font', 'Times', 'base_size', 16);
g.stat_boxplot('width',0.8);
g.set_names('x', 'Network Class', 'y', 'MSE');
g.draw();

save 'MSE_x4.mat' MSE_eh
save 'MSSIM_x4.mat' MSSIM_eh
save 'PSNR_x4.mat' PSNR_eh
%% MSE error bar
figure(4),
bar_x = repmat(1:8,1,36);
%color_sel = repmat({'x2','x3','x4'},1,size(MSE_eh, 2)*size(MSE_eh, 1)/3);
color_sel = repmat({'x2'},1,size(MSE_eh, 2)*size(MSE_eh, 1));
temp=reshape(MSE_eh, 1, size(MSE_eh, 1)*size(MSE_eh, 2));
temp_X = bar_x;
g=gramm('x', temp_X,'y',temp,'color', color_sel');%(MSE_bar_y);
g.geom_point('alpha',0.3,'dodge',0.1);
g.set_color_options('map','matlab');
g.set_text_options( 'font', 'Times', 'base_size', 16);
g.stat_boxplot('width',0.5);
g.set_names('x', 'Network Class', 'y', 'PSNR');
g.draw();
% bar_x = 1:idy;
% 
% boxplot(PSNR_eh');%(PSNR_bar_y);
% hold on
% er = errorbar(bar_x, PSNR_bar_y,abs(PSNR_el),abs(PSNR_eh));  
% 
% er.Color = [0 0 0];                            
% er.LineStyle = 'none';  
%xticklabels(ary)
%% PSNR error bar
figure(2),
bar_x = repmat(1:8,1,36);
%color_sel = repmat({'x2','x3','x4'},1,size(MSE_eh, 2)*size(MSE_eh, 1)/3);
color_sel = repmat({'x2'},1,size(PSNR_eh, 2)*size(PSNR_eh, 1));
temp=reshape(PSNR_eh, 1, size(PSNR_eh, 1)*size(PSNR_eh, 2));
temp_X = bar_x;
g=gramm('x', temp_X,'y',temp,'color', color_sel');%(MSE_bar_y);
g.geom_point('alpha',0.3,'dodge',0.1);
g.set_color_options('map','matlab');
g.set_text_options( 'font', 'Times', 'base_size', 16);
g.stat_boxplot('width',0.5);
g.set_names('x', 'Network Class', 'y', 'PSNR');
g.draw();
% bar_x = 1:idy;
% 
% boxplot(PSNR_eh');%(PSNR_bar_y);
% hold on
% er = errorbar(bar_x, PSNR_bar_y,abs(PSNR_el),abs(PSNR_eh));  
% 
% er.Color = [0 0 0];                            
% er.LineStyle = 'none';  
%xticklabels(ary)
%% MSSIM error barPSNR_eh
figure(3),
bar_x = repmat(1:8,1,36);
%color_sel = repmat({'x2','x3','x4'},1,size(MSE_eh, 2)*size(MSE_eh, 1)/3);
color_sel = repmat({'x2'},1,size(MSSIM_eh, 2)*size(MSSIM_eh, 1));
temp=reshape(MSSIM_eh, 1, size(MSSIM_eh, 1)*size(MSSIM_eh, 2));
temp_X = bar_x;
g=gramm('x', temp_X,'y',temp,'color', color_sel');%(MSE_bar_y);
g.geom_point('alpha',0.3,'dodge',0.1);
g.set_color_options('map','matlab');
g.set_text_options( 'font', 'Times', 'base_size', 16);
g.stat_boxplot('width',0.5);
g.set_names('x', 'Network Class', 'y', 'MSSIM');
g.draw();
% bar_x = 1:idy;
% 
% boxplot(MSSIM_eh');%(MSSIM_bar_y);
% hold on
% er = errorbar(bar_x, MSSIM_bar_y,abs(MSSIM_el),abs(MSSIM_eh));  
% 
% er.Color = [0 0 0];                            
% er.LineStyle = 'none';  
%xticklabels(ary)

%% x2x3x4 error bar
cd C:\Users\Mars\Desktop\ENLARGED\FORPLOT\PLOTDIVIDE
load MSE_x2.mat 
MSE_x2=MSE_eh;
load MSSIM_x2.mat 
MSSIM_x2=MSSIM_eh;
load PSNR_x2.mat 
PSNR_x2 = PSNR_eh;

clear MSE_eh MSSIM_eh PSNR_eh

load MSE_x3.mat
MSE_x3=MSE_eh;
load MSSIM_x3.mat
MSSIM_x3=MSSIM_eh;
load PSNR_x3.mat
PSNR_x3 = PSNR_eh;

clear MSE_eh MSSIM_eh PSNR_eh

load MSE_x4.mat
MSE_x4=MSE_eh;
load MSSIM_x4.mat
MSSIM_x4=MSSIM_eh;
load PSNR_x4.mat
PSNR_x4 = PSNR_eh;

figure,
ary2 = ary;%{'Bicubic','DBPN','HR','LR','ASR','ResNet','SRUnet'} ;
bar_x = repmat(ary2,1,3*30);
c0 = repmat({'2x'},1,size(MSE_eh, 2)*size(MSE_eh, 1));
c1 = repmat({'3x'},1,size(MSE_eh, 2)*size(MSE_eh, 1));
c2 = repmat({'4x'},1,size(MSE_eh, 2)*size(MSE_eh, 1));
%c2 = repmat({'x4'},1,size(MSSIM_eh, 2)*size(MSSIM_eh, 1));
color_sel = [c0,c1,c2];%[c0,c1,c2];
temp_X = bar_x;
temp=reshape([MSE_x2,MSE_x3,MSE_x4], 1, 3*size(MSE_x2, 1)*size(MSE_x2, 2));
g=gramm('x', temp_X,'y',temp,'color', color_sel);%(MSE_bar_y);
g.axe_property('XGrid','on','YGrid','on');
g.geom_point('alpha',0.3,'dodge',0.7);
g.set_color_options('map','brewer_pastel');
g.set_text_options( 'font', 'Times', 'base_size', 16);
g.stat_boxplot('width',0.5);
g.set_names('x', 'Network Class', 'y', 'MSE','color',{'Scale'});
%g.set_layout_options('position',[100 100 500 100]);
g.draw();

figure,
temp=reshape([PSNR_x2,PSNR_x3,PSNR_x4], 1, 3*size(MSE_eh, 1)*size(MSE_eh, 2));
g=gramm('x', temp_X,'y',temp,'color', color_sel);%(MSE_bar_y);
g.axe_property('XGrid','on','YGrid','on');
g.geom_point('alpha',0.3,'dodge',0.7);
g.set_color_options('map','brewer_pastel');
g.set_text_options( 'font', 'Times', 'base_size', 16);
g.stat_boxplot('width',0.5);
g.set_names('x', 'Network Class', 'y', 'PSNR','color',{'Scale'});
g.draw();

figure,
temp=reshape([MSSIM_x2,MSSIM_x3,MSSIM_x4], 1, 3*size(MSE_eh, 1)*size(MSE_eh, 2));
g=gramm('x', temp_X,'y',temp,'color', color_sel);%(MSE_bar_y);
g.axe_property('XGrid','on','YGrid','on');
g.geom_point('alpha',0.3,'dodge',0.7);
g.set_color_options('map','brewer_pastel');
g.set_text_options( 'font', 'Times', 'base_size', 16);
g.stat_boxplot('width',0.5);
g.set_names('x', 'Network Class', 'y', 'MSSIM','color',{'Scale'});
g.draw();