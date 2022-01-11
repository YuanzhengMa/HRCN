figure(1)
grid on 
sum_PSNR_ASBN = (sum_PSNR_ASR2+sum_PSNR_ASR2)/2;
sum_PSNR_DBPN = (sum_PSNR_DBPN1 +sum_PSNR_DBPN_OK)/2;
sum_PSNR_SRUNet = (sum_PSNR_SRUNet2 +sum_PSNR_SRUNet2)/2;
sum_PSNR_ResNet34 = (sum_PSNR_ResNet_OK +sum_PSNR_ResNet_OK)/2;

f1 = fit(sum_PSNR_ASBN',sum_PSNR_Input','poly1');
plot(f1,'r-',sum_PSNR_ASBN',sum_PSNR_Input','r.');
hold on 
f2 = fit(sum_PSNR_DBPN',sum_PSNR_Input','poly1');
plot(f2,'g-',sum_PSNR_DBPN',sum_PSNR_Input','g.');
hold on
f3 = fit(sum_PSNR_SRUNet', sum_PSNR_Input','poly1');
plot(f3,'b-', sum_PSNR_SRUNet', sum_PSNR_Input','b.');

hold on
f4 = fit(sum_PSNR_ResNet34', sum_PSNR_Input','poly1');
plot(f4,'c-', sum_PSNR_ResNet34', sum_PSNR_Input','c.');

legend({'data point_{ASBN}','fitted curve_{ASBN}','data point_{DBPN}','fitted curve_{DBPN}'...
    'data point_{SRUNet}','fitted curve_{SRUNet}','data point_{SRResNet}','fitted curve_{SRResNet}'})
title('PSNR')

%
figure(2)

sum_MSE_ASBN = (sum_MSE_ASR2+sum_MSE_ASR2)/2;
sum_MSE_DBPN = (sum_MSE_DBPN1 +sum_MSE_DBPN_OK)/2;
sum_MSE_SRUNet = (sum_MSE_SRUNet2 +sum_MSE_SRUNet2)/2;
sum_MSE_ResNet34 = (sum_MSE_ResNet_OK +sum_MSE_ResNet_OK)/2;

f1 = fit(sum_MSE_ASBN',sum_MSE_Input','poly1');
plot(f1,'r-',sum_MSE_ASBN',sum_MSE_Input','r.');
hold on 
f2 = fit(sum_MSE_DBPN',sum_MSE_Input','poly1');
plot(f2,'g-',sum_MSE_DBPN',sum_MSE_Input','g.');
hold on
f3 = fit(sum_MSE_SRUNet', sum_MSE_Input','poly1');
plot(f3,'b-', sum_MSE_SRUNet', sum_MSE_Input','b.');

hold on
f4 = fit(sum_MSE_ResNet34', sum_MSE_Input','poly1');
plot(f4,'c-', sum_MSE_ResNet34', sum_MSE_Input','c.');

legend({'data point_{ASBN}','fitted curve_{ASBN}','data point_{DBPN}','fitted curve_{DBPN}'...
    'data point_{SRUNet}','fitted curve_{SRUNet}','data point_{SRResNet}','fitted curve_{SRResNet}'})
title('MSE')
%
figure(3)

sum_MSSIM_ASBN = (sum_MSSIM_ASR2+sum_MSSIM_ASR2)/2;
sum_MSSIM_DBPN = (sum_MSSIM_DBPN1 +sum_MSSIM_DBPN_OK)/2;
sum_MSSIM_SRUNet = (sum_MSSIM_SRUNet2 +sum_MSSIM_SRUNet2)/2;
sum_MSSIM_ResNet34 = (sum_MSSIM_ResNet_OK +sum_MSSIM_ResNet_OK)/2;

f1 = fit(sum_MSSIM_ASBN',sum_MSSIM_Input','poly1');
plot(f1,'r-',sum_MSSIM_ASBN',sum_MSSIM_Input','r.');
hold on 
f2 = fit(sum_MSSIM_DBPN',sum_MSSIM_Input','poly1');
plot(f2,'g-',sum_MSSIM_DBPN',sum_MSSIM_Input','g.');
hold on
f3 = fit(sum_MSSIM_SRUNet', sum_MSSIM_Input','poly1');
plot(f3,'b-', sum_MSSIM_SRUNet', sum_MSSIM_Input','b.');

hold on
f4 = fit(sum_MSSIM_ResNet34', sum_MSSIM_Input','poly1');
plot(f4,'c-', sum_MSSIM_ResNet34', sum_MSSIM_Input','c.');

legend({'data point_{ASBN}','fitted curve_{ASBN}','data point_{DBPN}','fitted curve_{DBPN}'...
    'data point_{SRUNet}','fitted curve_{SRUNet}','data point_{SRResNet}','fitted curve_{SRResNet}'})
title('MSSIM')
 