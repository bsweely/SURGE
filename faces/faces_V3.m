clear; close all; clc;

load('Jeremy_data_2_98_.mat');
es = length(images);
Fs = 15;
roi = cell(Fs,1);
roiC = cell(es,1);
r = zeros(es,1);
g = zeros(es,1);
b = zeros(es,1);

tic
for i = 1:es
    img = images(i);
    img = img.snapshot;
    roiC{i} = detectbothcheeks_V3(img);
    roif = roiC{i}{1};
    roil = roiC{i}{2};
    roir = roiC{i}{3};
    if roif==1
        if i>1 
            roif = roiC{i-1}{1};
            roil = roiC{i-1}{2};
            roir = roiC{i-1}{3};
        else 
            roif = [250,150;300,250;250,250;300,150];
            roil = [250,150;300,250;250,250;300,150];
            roir = [250,150;300,250;250,250;300,150];
            roiC{i}{1} = [250,150;300,250;250,250;300,150];
            roiC{i}{2} = [250,150;300,250;250,250;300,150];
            roiC{i}{3} = [250,150;300,250;250,250;300,150];
        end
    end
    froi = img(round(roif(2,2)):round(roif(1,2)), round(roif(1,1)):round(roif(3,1)),:);
    lroi = img(round(roil(2,2)):round(roil(1,2)), round(roil(1,1)):round(roil(3,1)),:);
    rroi = img(round(roir(2,2)):round(roir(1,2)), round(roir(1,1)):round(roir(3,1)),:);
    
    fsum = sum(sum(froi,2),1);
    lsum = sum(sum(lroi,2),1);
    rsum = sum(sum(rroi,2),1);
    xx = [fsum;lsum;rsum];
    avg = sum(xx,1);
    
    r(i) = avg(1);
    g(i) = avg(2);
    b(i) = avg(3);
end
t = tic; % stop timer
time = toc(t); % time elapsed
fps = es/time; % calculated frame rate

t = 1:1/fps:time; % time series

% detrend
% normalize
r_detrend = detrend(r);
b_detrend = detrend(b);
g_detrend = detrend(g);

r_norm = normalize(r_detrend)';
b_norm = normalize(b_detrend)';
g_norm = normalize(g_detrend)';

% Best component selection
X = [r_norm; b_norm; g_norm];
[pulse_ica, W, T, mu] = kICA(X,3); % changed to 3 source, find best PPG signal
ica_s1 = pulse_ica(1,:);
ica_s2 = pulse_ica(2,:);
ica_s3 = pulse_ica(3,:);

% FFT to select which ICA source to use for further filtering 
N = length(pulse_ica); % same as es 
freq = 0:Fs/N:Fs/2;

source1_fft = fft(ica_s1); % fft of each source 
s1 = abs(source1_fft/N); 
s1_power_fft = s1(1:N/2+1).^2; 
s1_power_fft(2:end-1) = 2*s1_power_fft(2:end-1); 

source2_fft = fft(ica_s2); 
s2 = abs(source2_fft/N); 
s2_power_fft = s2(1:N/2+1).^2; 
s2_power_fft(2:end-1) = 2*s2_power_fft(2:end-1); 

source3_fft = fft(ica_s3); 
s3 = abs(source3_fft/N); 
s3_power_fft = s3(1:N/2+1).^2; 
s3_power_fft(2:end-1) = 2*s3_power_fft(2:end-1); 

figure(1) % Select the ICA source with the highest peak within the normal HR freq (0.8 to 3 in the freq domain, 48-180 bpm in the time domain)) 
subplot(3,1,1)
plot(freq, s1_power_fft)
subplot(3,1,2)
plot(freq, s2_power_fft)
subplot(3,1,3)
plot(freq, s3_power_fft)

freq_range = find(freq>=.8 & freq<=3) ; % extracts the indices of the frequencies that are within the range of 0.8 and 3 
source_select_array = [max(s1_power_fft(freq_range)), max(s2_power_fft(freq_range)), max(s3_power_fft(freq_range))]; % finds the max of each source of the corresponding indices of the frequencies between 0.8 and 3 
[source, source_number] = max(source_select_array); % finds which source has the max in the freq range compared to the others 

if source_number == 1 % determines which source will be the 'data' that is process through the rest of the code , should only have to happen once with each moving interval 
    data = ica_s1 ; % can follow the variable name 'data' through processing 
elseif source_number == 2 
    data = ica_s2 ;
else
    data = ica_s3 ;
end 
    

% Moving Average
data_mov_avg = smoothdata(data, 'movmean', 5); % trial moving avg function 'smoothdata'

figure (2) % before and after moving average to display smoothing of data 
plot(1:N, data,'r') 
hold on 
plot(1:N, data_mov_avg,'g')
    
% Bandpass Filter
[data_bp_filter,d]=bandpass(data,[0.8 3],Fs);

% FFT to find HR Freq 
data_fft = fft(data_bp_filter); 
P2 = abs(data_fft/N); 
data_power_fft = P2(1:N/2+1) ; 
data_power_fft(2:end-1) = 2*data_power_fft(2:end-1); 

figure (3) 
plot (freq, data_power_fft,'b') 

% peak detector
[maxPeak, index] = max(data_power_fft); % getting max peak and index
HR_avg = freq(index) * 60 ; 

% test peak detector
[peaks, freqs] = findpeaks(data_power_fft, freq);

HR = freqs.*60



























