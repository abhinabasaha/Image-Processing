clear all
close all

I=('tumor.jpg');
JJ=imread(I);

I_bw=rgb2gray(JJ);

%otsu's method threshold
[level EM] = graythresh(JJ);
BW = im2bw(JJ,level);

subplot(2,2,1);imshow(JJ);title('original image');
subplot(2,2,2);imshow(I_bw);title('grayscale image');
subplot(2,2,3);imshow(BW);title('binary image(by otsu method)');
subplot(2,2,4);imhist(I_bw);title('HISTOGRAM');

[h,locs]=imhist(I_bw);

[rows,coloumns]=size(I_bw);

image_bw=reshape(I_bw,369*400,1);
figure
imhist(I_bw);title('HISTOGRAM');
alpha=2;
 
imageI=reshape(JJ,369*400,3);

[center,U,objFcn]=fcm(double(imageI),5);

U=smf(U,[0 1]);
%plot(U)

U1=U(1,:);
Ureshp1=reshape(U1,369,400);

U2=U(2,:);
Ureshp2=reshape(U2,369,400);

U3=U(3,:);
Ureshp3=reshape(U3,369,400);

U4=U(4,:);
Ureshp4=reshape(U4,369,400);

U5=U(5,:);
Ureshp5=reshape(U5,369,400);

subplot(2,3,1);imshow(Ureshp1);title('Cluster 1');
subplot(2,3,2);imshow(Ureshp2);title('Cluster 2');
subplot(2,3,3);imshow(Ureshp3);title('Cluster 3');
subplot(2,3,4);imshow(Ureshp4);title('Cluster 4');
subplot(2,3,5);imshow(Ureshp5);title('Cluster 5');

maxU=max(U);
figure,plot(U);
% for i=1:147600
%     index(1,i)=find(U(:,i) == maxU);
% end
index1 = find(U(1,:) == maxU);
index2 = find(U(2,:) == maxU);
index3 = find(U(3,:) == maxU);
index4 = find(U(4,:) == maxU);
index5 = find(U(5,:) == maxU);
clustersoln(index1)=1;
clustersoln(index2)=2;
clustersoln(index3)=3;
clustersoln(index4)=4;
clustersoln(index5)=5;
clustered_image=reshape(clustersoln,369,400);

plot(clustered_image);title('PLOT OF CLUSTERED IMAGE');
% index1 = find(U(1,:) == maxU);
% index2 = find(U(2,:) == maxU);
% index1=index1.';
% 
% plot(double(imageI)(index1,1),fcmdata(index1,2),'oc')
% hold on
% plot(double(imageI(index2,1),fcmdata(index2,2),'or')
% plot(centers(1,1),centers(1,2),'xb','MarkerSize',15,'LineWidth',3)
% plot(centers(2,1),centers(2,2),'xr','MarkerSize',15,'LineWidth',3)
% hold off


for i=0:255
    clear rangei;
    rangei=find(image_bw==i);
    murange(i+1)=sum(maxU(rangei));
    
end
x=1:1:400;
mem_func=smf(x,[1 400]);
%memf=mem_func(U);

%memf=smf(murange,[0 1]);
 memf=smf(murange,[min(murange) max(murange)]);
 pixelmf=smf(maxU,[min(maxU) max(maxU)]);
 figure
plot(memf);title('membership function');

%memf=murange;
    mu_u=memf.^(1/alpha);
    mu_l=memf.^(alpha);
    H=transpose(h);
    a=mu_u-mu_l;
    
k=1/(369*400);
b=H.*a;

gamma=k*b;

% b(:);
% [M,I]=max(b(:))
% [I_row,I_col]=ind2sub(size(b),I)

gamma(:);
[M,I]=max(gamma(:))
[I_row,I_col]=ind2sub(size(gamma),I)

% for i=1:1:400
%     if memf(i)<M
%         ambi(i)= memf(i)
%     end
% end

ambi=find(pixelmf<M);
figure,imshow(ambi,'displayRange');

%[rowsP,colomnsP]=size(pixelmf);

Reshapedimage=reshape(pixelmf,369,400);
IM2 = imcomplement(Reshapedimage);
subplot(1,2,1);imshow(Reshapedimage);title('RESHAPED IMAGE');
subplot(1,2,2);imshow(IM2);title('COMPLIMENT');

ambi_loc=find(Reshapedimage<M);

ppp=Reshapedimage(ambi_loc);

for k=1:1:187
 for g=1:1:369
    for h=1:1:400
      if ppp(k)==Reshapedimage(g,h)
          wind(k,:)=[clustered_image(g-1,h-1),clustered_image(g,h-1),clustered_image(g+1,h-1),clustered_image(g-1,h),clustered_image(g,h),clustered_image(g+1,h),clustered_image(g-1,h+1),clustered_image(g,h+1),clustered_image(g+1,h+1)];
          %wind1(k,:)=[Reshaped]
          prev_cluster(k,1)=clustered_image(g-1,h-1);
      end
    end
 end
end


[the_no,Freq,C] = mode(wind,2);


 ReCluster=reshape(clustered_image,147600,1);

 
 for i=1:1:187
     j=ambi_loc(i);
     ReCluster(j)=the_no(i);
 end
 
 clust_img_ambiAssign=reshape(ReCluster,369,400);
 figure
 imshow(clust_img_ambiAssign, 'displayRange',[]),title('result of proposed approach');
 
 % %%%%%%%%%%%%%%%%%%%%  CLOSING  %%%%%%%%%%%%%%%%%%%%
% se=strel('diamond',4);
% cldia=imclose(Ureshp2,se);
% se=strel('disk',4);
% cldisk=imclose(Ureshp2,se);
% se=strel('octagon',6);
% cloct=imclose(Ureshp2,se);
% se=strel('square',4);
% clsq=imclose(Ureshp2,se);
% figure,
%  subplot(2,2,1);imshow(cldia);title('closing by diamond');
%  subplot(2,2,2);imshow(cldisk);title('closing by disk');
%  subplot(2,2,3);imshow(cloct);title('closing by octagon');
%  subplot(2,2,4);imshow(clsq);title('closing by square');
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %  %%%%%%%%%%%%%%%%%%% OPENING  %%%%%%%%%%%%%%%%%%%%%%%
%  se=strel('diamond',4);
% opdia=imopen(Ureshp2,se);
% se=strel('disk',4);
% opdisk=imopen(Ureshp2,se);
% se=strel('octagon',6);
% opoct=imopen(Ureshp2,se);
% se=strel('square',4);
% opsq=imopen(Ureshp2,se);
% figure,
%  subplot(2,2,1);imshow(cldia);title('opening by diamond');
%  subplot(2,2,2);imshow(cldisk);title('opening by disk');
%  subplot(2,2,3);imshow(cloct);title('opening by octagon');
%  subplot(2,2,4);imshow(clsq);title('opening by square');
% % 
% %  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %  
% %  %%%%%%%%%%%%%%%%%% ERODE %%%%%%%%%%%%%%%%%%%%%%%%
% se=strel('diamond',4);
% erdia=imerode(Ureshp2,se);
% se=strel('disk',4);
% erdisk=imerode(Ureshp2,se);
% se=strel('octagon',6);
% eroct=imerode(Ureshp2,se);
% se=strel('square',4);
% ersq=imerode(Ureshp2,se);
% figure,
%  subplot(2,2,1);imshow(erdia);title('erode by diamond');
%  subplot(2,2,2);imshow(erdisk);title('erode by disk');
%  subplot(2,2,3);imshow(eroct);title('erode by octagon');
%  subplot(2,2,4);imshow(ersq);title('erode by square');
% % 
% %  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %  
% %  %%%%%%%%%%%%%%%% DILATE %%%%%%%%%%%%%%%%%%%%%
% se=strel('diamond',4);
% dildia=imdilate(Ureshp2,se);
% se=strel('disk',4);
% dildisk=imdilate(Ureshp2,se);
% se=strel('octagon',6);
% diloct=imdilate(Ureshp2,se);
% se=strel('square',4);
% dilsq=imdilate(Ureshp2,se);
% figure,
%  subplot(2,2,1);imshow(dildia);title('dilate by diamond');
%  subplot(2,2,2);imshow(dildisk);title('dilate by disk');
%  subplot(2,2,3);imshow(diloct);title('dilate by octagon');
%  subplot(2,2,4);imshow(dilsq);title('dilate by square');
% % 
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  
%  %%%%%%%%%%%%%%%%%%%%% EDGE DETECTION %%%%%%%%%%%%%%%%%
 BBW1 = edge(Ureshp2,'Sobel','nothinning'); 
 BBW2 = edge(Ureshp2,'Prewitt','nothinning');
 BBW3 = edge(Ureshp2,'Roberts','nothinning');
 BBW4 = edge(Ureshp2,'log');
 figure,
 subplot(2,2,1);imshow(BBW1);title('sobel');
 subplot(2,2,2);imshow(BBW1);title('Prewitt');
 subplot(2,2,3);imshow(BBW1);title('Roberts');
 subplot(2,2,4);imshow(BBW1);title('log');
%   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
%  figure
%  Result=clust_img_ambiAssign('')
 
 
% data = reshape(ReCluster,1,147600);
% figure;
% hold('all');
% MarkerOrder = '+o*x';
% MarkerIndex = 1;
% ColorOrder  = lines(147600);  % see "doc colormap"
% ColorIndex  = 1;
% H = zeros(1, 147600);  % Store handles
% for i = 1:147600
%   H(i) = plot(i, data(i), ...
%               'Marker', MarkerOrder(MarkerIndex), ...
%               'Color', ColorOrder(ColorIndex, :));
%   ColorIndex = ColorIndex + 1;
%   if ColorIndex > size(ColorIndex, 1)
%     ColorIndex  = 1;
%     MarkerIndex = MarkerIndex + 1;
%     if MarkerIndex > size(MarkerIndex, 2)
%        MarkerIndex = 1;
%     end
%   end
% end
% title('CLUSTERS'); xlabel('NO. OF PIXELS-->');ylabel('CLUSTERS-->');
% %legend(H, {'Name1', 'Name2', 'Name3','Name4','Name5','Name6','Name7','Name8','Name9','Name10','Name11','Name12','Name13','Name14','Name15','Name16'})

 
 