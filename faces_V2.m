clear; close all; clc;

Fs = 15; % framerate needs to be higher 
% home
mypi=raspi('IP_Address','pi','password');
cam = cameraboard(mypi,'Resolution','640x480','FrameRate',Fs,'Quality',50);
end_sample=20; % set how many seconds you want to loop
es = 20;
roi = cell(Fs,1);

tstart = tic;
for i = 1:es 
    img = snapshot(cam);
    if mod(i,5) == 0  
    roi{i} = detectfaces_V2(img);
    else
        roi{i} = roi{i-1}; 
    end 
    if roi{i}==1
        if i>1 
            roi{i} = roi{i-1};
        else 
            roi{i} = [250,150;300,250;250,250;300,150];
        end
    end
    x = roi{i}(:,2);
    y = roi{i}(:,1);
    Red_ROI = img(min(x):max(x),min(y):max(y),1);
    Blue_ROI = img(min(x):max(x),min(y):max(y),3);
    Green_ROI = img(min(x):max(x),min(y):max(y),2);
    r(i) = sum(sum(Red_ROI)); % intensity -> PPG
    b(i) = sum(sum(Blue_ROI));
    g(i) = sum(sum(Green_ROI));
end
time = toc(tstart); % display time in seconds 
fps = es/time; % calculated frame rate 

t_series = linspace(0,time,es); % time series the same length as RGB signals


% detrend
r_detrend = detrend(r);
b_detrend = detrend(b);
g_detrend = detrend(g);

% normalize
r_norm = normalize(r_detrend);
b_norm = normalize(b_detrend);
g_norm = normalize(g_detrend);

% ICA feature selection OR PCA
X = [t_series; r_norm; b_norm; g_norm];
[pulse_ica, W, T, mu] = kICA(X,3); % changed to 3 source, find best PPG signal

% Power Spectral Density to select which component to use
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

% Best component selection
best_comp = 3; % green channel
Xb = X([1 3],:);

% Moving Average
sw_size = 5; % window size
pulse_sw = zeros(length(Xb),1);

for ii = 1:length(psdx)
    if ii > sw_size
          pulse_sw(ii) = mean(Xb(ii-sw_size:ii));
    end
end

% Bandpass Filter
[filter_out,d]=bandpass(pulse_sw(sw_size+1:end),[0.5 5],Fs);
Y=filter_out(1:length(Xb)-sw_size);
pulse_fft = fft(Y) ; 
P2 = abs(pulse_fft/N); 
power_fft = P2(1:N/2+1) ; 
power_fft(2:end-1) = 2*power_fft(2:end-1); 

figure(2)
pspectrum(power_fft,Fs) 

figure (3) 
plot (freq, 10*log10(power_fft)) 

% peak detector
[peaks,locs] = findpeaks(10*log10(power_fft), freq);

HR = locs(1,1) * 60 ;

% plot data
figure(4)
plot(1:end_sample, r_sliding_window_avg, 'r');
hold on
plot(1:end_sample, g_sliding_window, 'g');
plot(1:end_sample, b_sliding_window_avg, 'b');
hold off
figure(5)
plot(1:Fs, pulse_ica(1,:), 'r')
hold on
plot(1:Fs, pulse_ica(3,:), 'g')
plot(1:Fs, pulse_ica(2,:), 'b')

% Save Data
save('data_Initials_video#.mat','r','b','g');
