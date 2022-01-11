clear all
clc
T = ["Bicubic","GT","ResNet","Input","HRCN","ResNetNoise"];
path = "C:\Users\Administrator\Desktop\need\for_pdist\";
eval(strcat("D1 = dir('",path,"');"));
MARKER = ['b.';'r.';'y.';'g.';'m.';'k.'];
for idx = 3:numel(D1)
    subpath = strcat(path,D1(idx).name);
    for idy = 1:10
        i1=imread(strcat(subpath,'\',num2str(idy),'.jpg'));
        if idx == 3
            i1 = imresize(i1,2,"bicubic");
        end
        i1=i1(:,:,1);
        [c1,n]=imhist(i1);
        c1=c1/size(i1,1)/size(i1,2);
        i2=imread(strcat(path,"GT","\",num2str(idy),'.jpg'));
        i2=i2(:,:,1);
        [c2,n2]=imhist(i2);
        c2=c2/size(i2,1)/size(i2,2);
        d=pdist2(c1',c2','euclidean');
        eval(strcat("dc","(",num2str(idx-2),",",num2str(idy),")=","d;"));
    end
    A = dc(idx-2,:);
    plot(A', MARKER(idx-2,:),'LineWidth',2,'MarkerSize',15)
    hold on
end
