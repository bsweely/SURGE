clear; close all; clc;

Fs = 10; % framerate needs to be higher 
% home
mypi=raspi('10.0.0.52','pi','password');
cam = cameraboard(mypi,'Resolution','640x480','FrameRate',Fs,'Quality',50);
end_sample=20; % set how many seconds you want to loop
es = 50; % If I set this to be es = 10, then it has a syntax error?
roi = cell(Fs,1);
rTotal = zeros(1,50);
bTotal = zeros(1,50);
gTotal = zeros(1,50);
HR = [];

% Looping over to make a moving average for heart rate
for j = 1:1 % Number of interations to get heart rate measurement
    for i = 1:es
    img = snapshot(cam);
    roi{i} = detectfaces_V2(img)
    if roi{i}==1
        if i>1 
            roi{i} = roi{i-1}
        else 
            roi{i} = [250,150;300,250;250,250;300,150];
        end
    end
    x = roi{i}(:,2)
    y = roi{i}(:,1)
    Red_ROI = img(min(x):max(x),min(y):max(y),1);
    Blue_ROI = img(min(x):max(x),min(y):max(y),3);
    Green_ROI = img(min(x):max(x),min(y):max(y),2);
    
    % Making sure that I am extracting the correct coordinates
    figure(10)
    imagesc(Red_ROI)
    figure(20)
    imagesc(Blue_ROI)
    figure(30)
    imagesc(Green_ROI)
    
    % Collecting intensities
    r(i) = sum(sum(Red_ROI)); % intensity -> PPG
    b(i) = sum(sum(Blue_ROI));
    g(i) = sum(sum(Green_ROI));
    end
% Save RGB data into the RGB arrays that serve data for the moving average
% calculation
rTotal = horzcat(rTotal, r);
bTotal = horzcat(bTotal, b);
gTotal = horzcat(gTotal, g);

% Removing all but the last 40 elements of each array
rTotal(1:(length(rTotal) - 40)) = [];
bTotal(1:(length(bTotal) - 40)) = [];
gTotal(1:(length(gTotal) - 40)) = [];

% detrend
r_detrend = detrend(rTotal);
b_detrend = detrend(bTotal);
g_detrend = detrend(gTotal);

% normalize
r_norm = normalize(r_detrend);
b_norm = normalize(b_detrend);
g_norm = normalize(g_detrend);

% ICA feature selection
X = [r_norm; b_norm; g_norm];
[pulse_ica, W, T, mu] = kICA(X,3); % changed to 3 source, find best PPG signal

% which data channel has most relevent signal to HR
% making the dataset with 

% Power Spectral Density
t = 0:1/Fs:1-1/Fs;
N = length(pulse_ica);
xdft = fft(pulse_ica);
xdft = xdft(1:N/2+1);
psdx = (1/(Fs*N)) * abs(xdft).^2;
psdx(2:end-1) = 2*psdx(2:end-1);
freq = 0:Fs/N:Fs/2;
yPSD = 10*log10(psdx);

figure(1)
plot(freq,10*log10(psdx))
grid on
title('Periodogram Using FFT')
xlabel('Frequency (Hz)')
ylabel('Power/Frequency (dB/Hz)')

% Moving Average
sw_size = 5; % window size
pulse_sw = zeros(length(psdx),1);

for ii = 1:length(psdx)
    if ii > sw_size
          pulse_sw(ii) = mean(psdx(ii-sw_size:ii));
    end
end

% Bandpass Filter
[filter_out,d]=bandpass(pulse_sw(sw_size+1:end),[0.5 5],Fs);
y=filter_out(1:length(psdx)-sw_size);
pulse_fft=abs(fft(y,length(y)));
figure(2)
pspectrum(pulse_fft,Fs) 

% peak detector
[peaks,locs] = findpeaks(yPSD, freq);
HR = horzcat(HR, 60.*locs);


%{
% plot data
figure(3)
plot(1:end_sample, r_sliding_window_avg, 'r');
hold on
plot(1:end_sample, g_sliding_window, 'g');
plot(1:end_sample, b_sliding_window_avg, 'b');
hold off
figure(4)
plot(1:Fs, pulse_ica(1,:), 'r')
hold on
plot(1:Fs, pulse_ica(3,:), 'g')
plot(1:Fs, pulse_ica(2,:), 'b')

% Save Data
save('data_Initials_video#.mat','r','b','g');

%}

end

HR

% Calculating how many heart rates are in the normal range in this dataset
normHRCount = 0;
for i = 1:length(HR)
    if HR(i) < 100 && HR(i) > 60
        normHRCount = normHRCount + 1;
    end
end

normHRPercentage = 100*normHRCount./length(HR)

figure(3)
x = 1:length(HR);
plot(x,HR);
xlabel('HR Number in list');
ylabel('HR');
title('HR values')






