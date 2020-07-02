clear; close all; clc;

Fs = 15; % framerate needs to be higher 
% home
mypi=raspi('10.0.0.52','pi','password');
cam = cameraboard(mypi,'Resolution','640x480','FrameRate',Fs,'Quality',50);
end_sample=20; % set how many seconds you want to loop
es = 20;
roi = cell(Fs,1);

for i = 1:es
    img = snapshot(cam);
    roi{i} = detectfaces_V2(img);
    if roi{i}==1
        if i>1 
            roi{i} = roi{i-1};
        else 
            roi{i} = [250,150;300,250;250,250;300,150];
        end
    end
    x = roi{i}(:,2);
    y = roi{i}(:,1);
    % Forehead-only Data
    % Online, multiple art websites and facial recognition sites said the
    % forehead starts about halfway up the face and ends 3/4 up the face. 
    % The forehead spans about the same width of the face, and even more so
    % for some people, so the x coordinates for the forehead ROI are the
    % same as the face, but the y coordinates are limited to the upper 3/4
    % of the face only.
    faceHeight = max(y) - min(y);
    bottomOfFH = round(min(y) + faceHeight*0.5); % approximately the height of bottom of forehead
    topOfFH = round(min(y) + faceHeight*0.75); % approximately the height of top of forehead
    
    Red_ROI_FH = img(min(x):max(x),bottomOfFH:topOfFH,1);
    Blue_ROI_FH = img(min(x):max(x),bottomOfFH:topOfFH,3);
    Green_ROI_FH = img(min(x):max(x),bottomOfFH:topOfFH,2);
    rFH(i) = sum(sum(Red_ROI_FH));
    bFH(i) = sum(sum(Blue_ROI_FH));
    gFH(i) = sum(sum(Green_ROI_FH));
end

% detrend
r_detrend = detrend(rFH);
b_detrend = detrend(bFH);
g_detrend = detrend(gFH);

% normalize
r_norm = normalize(r_detrend);
b_norm = normalize(b_detrend);
g_norm = normalize(g_detrend);

% ICA feature selection
X = [r_norm; b_norm; g_norm];
[pulse_ica, W, T, mu] = kICA(X,3); % changed to 3 source, find best PPG signal

% which data channel has most relevent signal to HR

% Power Spectral Density
t = 0:1/Fs:1-1/Fs;
N = length(pulse_ica);
xdft = fft(pulse_ica);
xdft = xdft(1:N/2+1);
psdx = (1/(Fs*N)) * abs(xdft).^2;
psdx(2:end-1) = 2*psdx(2:end-1);
freq = 0:Fs/N:Fs/2;

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
%}

% Save Data
% save('data_Initials_video#.mat','r','b','g');