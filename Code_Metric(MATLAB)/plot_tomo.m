clc
clear all
cd 
path = ' C:\Users\Administrator\Desktop\need\fig6\plot_tomo2\'%' C:\Users\Mars\Desktop\processed\plot_tomox4\';
eval(strcat('cd',path));
D = dir(strcat(path(2:end), '*.mat'));
%D = dir(strcat(path(2:end), '*.png'));
C = {};
for idx = 1:numel(D)
    C(idx) = cellstr(D(idx).name);
end
%C = natsort(C);

f1 = figure(1);
axes1 = axes('Parent',f1);
legend_name = {};
color = {'#7E2F8E';'k';'#FFFF00';'r';'#00FF00';'#EDB120';'#4DBEEE'};
%color = {'r';'r';'r';'r';'r';'r';'r'};
for  idx  = 1:numel(C)
    name_col =  split(C(idx),'_');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    name = char(name_col(3));
    %eval(strcat('temp = imread(char(C(idx)));'));
    eval(strcat("load ", char(C(idx)), ";"));
    if  length(size(temp)) == 3
        eval(strcat(name,'=rgb2gray(temp);'));
    else
        eval(strcat(name,'=temp;'));
    end
    if idx == 1
        eval(strcat('L = size(', name, ',  1);'));
    end
    if strcmp(name(1:3), 'Inp')
       eval(strcat(name,' = imresize(', name,',size(GroundTrue),''nearest'');'));
    end
    legend_name(idx) = cellstr(name);
    %if strcmp(name, 'Bicubic')
%        eval(strcat('plot(', name, '(176,226:295), ''MarkerEdgeColor'',char(color(idx)),''LineWidth'',3);'));%(193,181:250)
%     elseif strcmp(name, 'Res3Net')
%         eval(strcat('plot(', name, '(193,181:250), ''MarkerEdgeColor'',char(color(idx)),''LineWidth'',1.5);'));
    %else
        %eval(strcat('plot(', name, '(193,181:250), ''MarkerEdgeColor'',char(color(idx)),''LineWidth'',1.5,''LineStyle'',''--'');'));
        eval(strcat('plot(', name, '(190,221:300), ''Color'',char(color(idx)),''LineWidth'',3);'));%Row2 GT(176,211:280)fei(193,181:250)
    %end
    ylim([-160,160]);
    hold on
end
legend(legend_name);

set(gca, 'FontSize', 20);
legend1 = legend(axes1,'show');
set(legend1,'FontSize',18);
xlabel('Epochs');
ylabel('Loss');
%原始的
%ylim([0,450]);
ylim([0,150]);
xlim([0,60]);
grid(axes1, 'on');
%set(f1,'position',[100,100,1500,500]);
set(axes1,'FontName','Times');
saveas(gca, strcat('Loss_train.eps'), 'psc');
saveas(gca, strcat('Loss_train.fig'));