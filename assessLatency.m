% Draft - analysis tool used to extract latency comparing hand and cursor
% motion
%
% Note this is a basic version - need to specify if video is vertical
% (options.vid_type = 'v') or horizontal (options.vid_type = 'h');
%
% Alkis Hadjiosif and Maurice Smith

function lat_stats = assessLatency(filename,options)

%Load data
readerobj = VideoReader(filename);

vid_type = options.vid_type; %horizontal or vertical
cursor_type = 'p'; %update this

v.numFrames = get(readerobj, 'NumberOfFrames');
v.FrameRate = get(readerobj, 'FrameRate');
v.h = get(readerobj, 'Height');
v.w = get(readerobj, 'Width');

N=10;
M = N*floor(v.numFrames/N);
KTHRESH = 400;

% calculate sum of video frames
vidFrames1 = read(readerobj,1+[0,N-1]);
im_sum = zeros(size(vidFrames1(:,:,:,1)));
intens_ = zeros(size(vidFrames1(:,:,:,1)));

for k=1:M
    % threshold before sum?
    if rem(k-1,N)==0, vidFrames1 = read(readerobj,k+[0,N-1]); end
    vidFrames=vidFrames1(:,:,:,rem(k-1,N)+1);

    im_sum = im_sum + double(vidFrames);
    intens_ = intens_ + sqrt(squeeze(sum(double(vidFrames).^2,3)));

end

% Select area to look for cursor vs. hand
figure;imagesc(uint8(intens_./(M/5)));
disp('Select cursor coords:');
h1 = imrect();
c = round(h1.getPosition);
c_xmin = c(1);
c_xmax = c(1) + c(3);
c_ymin = c(2);
c_ymax = c(2) + c(4);


disp('Select hand coords:');
h2 = imrect();
h = round(h2.getPosition);
h_xmin = h(1);
h_xmax = h(1) + h(3);
h_ymin = h(2);
h_ymax = h(2) + h(4);

%%
IThresh = 254;
% ic bounds (cursor)
KTHRESHx1 = c_ymin;
KTHRESHx2 = c_ymax;
KTHRESHy1 = c_xmin;
KTHRESHy2 = c_xmax;
% ip bounds (hand)
KTHRESHxx1 = h_ymin; % vertical bounds (rows)
KTHRESHxx2 = h_ymax;
KTHRESHyy1 = h_xmin;% horizontal bounds (cols)
KTHRESHyy2 = h_xmax;

if ishandle(999), close(999); end
figure(999);
set(999,'Position',[200 200 400 50]);
xlim([0 1]);
axis off;

for k=1:M
    if rem(k-1,N)==0, vidFrames1 = read(readerobj,k+[0,N-1]); end
    vidFrames=vidFrames1(:,:,:,rem(k-1,N)+1);

    figure(999);
    title(['Processing frame ', num2str(k),'/',num2str(M)]);
    if (mod(k,100)==0)
        barh(1,k/M,1,'facecolor',[0 0 1]);
        xlim([0 1]);
        axis off;
    end
    %shg;

    d = size(vidFrames);

    if cursor_type == 'p'
        red_ = double(vidFrames(:,:,1));
        green_ = double(vidFrames(:,:,2));
        blue_ = double(vidFrames(:,:,3));

        int_ = uint8(sqrt(red_.^2 + green_.^2 + blue_.^2));

        int_ = int_>IThresh;

        [ic{k}.x,ic{k}.y] = find(int_(KTHRESHx1:KTHRESHx2,KTHRESHy1:KTHRESHy2));
        [ip{k}.x,ip{k}.y] = find(int_(KTHRESHxx1:KTHRESHxx2,KTHRESHyy1:KTHRESHyy2));

    else
        red_ = vidFrames(:,:,1)>90;
        green_ = vidFrames(:,:,2)>160;
        blue_ = vidFrames(:,:,3)>150;
        if vid_type == 'h'
            [ic{k}.x,ic{k}.y] = find(red_(1:KTHRESH,:));
            [ip{k}.x,ip{k}.y] = find(blue_(KTHRESH:KTHRESH+500,:));
        else
            [ic{k}.x,ic{k}.y] = find(blue_(:,1:KTHRESH));
            [ip{k}.x,ip{k}.y] = find(blue_(:,KTHRESH:end));
        end
    end

    mean_ic.x(k)=mean(ic{k}.x);
    mean_ic.y(k)=mean(ic{k}.y);
    med_ic.x(k)=median(ic{k}.x);
    med_ic.y(k)=median(ic{k}.y);
    num_ic(k) = length(ic{k}.x);
    mean_ip.x(k)=mean(ip{k}.x);
    mean_ip.y(k)=mean(ip{k}.y);
    med_ip.x(k)=median(ip{k}.x);
    med_ip.y(k)=median(ip{k}.y);
    num_ip(k) = length(ip{k}.x);
end

figure; plot([mean_ic.x;mean_ip.x]');
legend('cursor', 'hand');
xlabel('frame number','FontSize', 24);
ylabel('position (pixels)','FontSize', 24);
title('Cursor + Hand Position Signals','FontSize', 24);

%figure; plot([mean_ic.y;med_ic.y;mean_ip.y;med_ip.y]');

%%
TTHRESH =300;
ii_ = ~isnan(mean_ic.y)&~isnan(mean_ip.y);
TMAX = sum(ii_);

if vid_type == 'h'

    ii_ = ~isnan(mean_ic.y)&~isnan(mean_ip.y);
    mean_ic.y= mean_ic.y(ii_);mean_ip.y = mean_ip.y(ii_);
    med_ic.y = med_ic.y(ii_);med_ip.y = med_ip.y(ii_);
    MM = numel(mean_ic.y);

    ii_ = ~isnan(mean_ic.y)&~isnan(mean_ip.y);
    mean_ic.y= mean_ic.y(ii_);mean_ip.y = mean_ip.y(ii_);
    med_ic.y = med_ic.y(ii_);med_ip.y = med_ip.y(ii_);

    a = interp1(TTHRESH:TMAX,mean_ic.y(TTHRESH:TMAX),TTHRESH:0.1:TMAX);
    b = interp1(TTHRESH:TMAX,mean_ip.y(TTHRESH:TMAX),TTHRESH:0.1:TMAX);

elseif vid_type=='v'

    ii_ = ~isnan(mean_ic.x)&~isnan(mean_ip.x);
    TMAX = sum(ii_); %Alkis: added this 2/29/2024
    mean_ic.x= mean_ic.x(ii_);mean_ip.x = mean_ip.x(ii_);
    med_ic.x = med_ic.x(ii_);med_ip.x = med_ip.x(ii_);
    MM = numel(mean_ic.x);

    a = interp1(TTHRESH:TMAX,mean_ic.x(TTHRESH:TMAX),TTHRESH:0.1:TMAX);
    b = interp1(TTHRESH:TMAX,mean_ip.x(TTHRESH:TMAX),TTHRESH:0.1:TMAX);

end
[cc,lags] = xcorr(a-mean(a),b-mean(b));
cc = cc./((norm(a-mean(a))*norm(-b+mean(b))));

figure;hold on;
plot(lags/v.FrameRate*100,cc);grid on; xlim([0 1200]);
xlabel('Lag steps (ms)'); ylabel('Normalized Hand-Cursor cross-correlation');
cc = cc(find(lags>0));
lags = lags(find(lags>0));
[~,i_ccmax] = max(cc);
delay = lags(i_ccmax)/v.FrameRate*100;title(['Delay = ',num2str(delay),' ms'])
plot([delay delay],[-1 1],'r--');
plot(delay,cc(i_ccmax),'r.','markersize',16);

lat_stats.EstimatedLatency = delay;
