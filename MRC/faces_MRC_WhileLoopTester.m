clear; close all; clc;

Fs = 30; % framerate needs to be higher % 480p@90fps is the max fps the camera data sheet specifies
% home

mypi=raspi('10.0.0.52','pi','password');
cam = cameraboard(mypi,'Resolution','640x480','FrameRate',Fs,'Quality',50);

end_sample=20; % set how many seconds you want to loop
es = 20;
roi = cell(Fs,1);
i = 0;

%{
% Test image files
path = strcat("C:\Users\jrsho\Desktop\myFacePictures");
files = dir(fullfile(path,"*"));
length_files = length(files)
%}

time = [];
HR = [];
while true
    i = i + 1;
    % length(HR) % used for debugging
    for i = i:(i+es-1)
        tic
        % img = imread(strcat(path,'\',files(i+2).name)); % I have +2 because that is when pictures start
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
        Red_ROI = img(min(x):max(x),min(y):max(y),1); 
        Green_ROI = img(min(x):max(x),min(y):max(y),2);
        Blue_ROI = img(min(x):max(x),min(y):max(y),3);
        r(i) = sum(sum(Red_ROI)); % intensity -> PPG
        g(i) = sum(sum(Green_ROI));
        b(i) = sum(sum(Blue_ROI));
        t = tic;
        time = horzcat(time, [toc(t)]);
    end
    t_series = linspace(0,toc(t),es); % time series the same length as RGB signals
    
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
    y=filter_out(1:length(Xb)-sw_size);
    pulse_fft=abs(fft(y,length(y)));
    figure(2)
    pspectrum(pulse_fft,Fs) 

    % peak detector
    [peaks,locs] = findpeaks(10*log10(pulse_fft), t); % [peaks,locs] = findpeaks(10*log10(pulse_fft), freq); % old code

    % Evaluating the heart rates
    HR = horzcat(HR, 60.*locs);
    HR;
    averageHR = mean(HR);
    
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
    
    HR;
    % Calculating how many heart rates are in the normal range in this dataset
    normHRCount = 0;
    for j = 1:length(HR)
        if HR(j) < 100 && HR(j) > 60
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
    %%%%%%%%%%%%%%%%%
    
    
    % Loop Frames
    if i == length(files)
        i = 0;
    end
end
    
% Save Data
% save('data_Initials_video#.mat','r','g','b');