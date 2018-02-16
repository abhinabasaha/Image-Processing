
close all

P=('tumor.jpg');
I=imread(P);
a=rgb2gray(I);
figure, imshow(a);title('grayscale');
imData=reshape(a,[],1);
imData=double(imData);
[IDX nn] = kmeans(imData,4);
imIDX=reshape(IDX,size(a));

figure, imshow(imIDX,[]),title('index image');
figure, 
  subplot(3,2,1),imshow(imIDX==1,[]);
  subplot(3,2,2),imshow(imIDX==2,[]);
  subplot(3,2,3),imshow(imIDX==3,[]);
  subplot(3,2,4),imshow(imIDX==4,[]);

bw1 = (imIDX==1);
se=ones(5);
bw1=imopen(bw1,se);
bw1=bwareaopen(bw1,400);

bw2 = (imIDX==2);
se=ones(5);
bw2=imopen(bw2,se);
bw2=bwareaopen(bw2,400);

bw3 = (imIDX==3);
se=ones(5);
bw3=imopen(bw3,se);
bw3=bwareaopen(bw3,400);

bw4 = (imIDX==4);
se=ones(5);
bw4=imopen(bw4,se);
bw4=bwareaopen(bw4,400);

figure,
  subplot(2,2,1),imshow(bw1),title('Cluster-1');
  subplot(2,2,2),imshow(bw2),title('Cluster-2');
  subplot(2,2,3),imshow(bw3),title('Cluster-3');
  subplot(2,2,4),imshow(bw4),title('Cluster-4');