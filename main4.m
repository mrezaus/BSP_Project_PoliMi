% =========================================================================
% SLEEP STAGE ANALYSIS (UNIFIED VERSION)
% Preprocessing & Hypnogram by MReza
% Welch PSD, Frequency Bands & Statistical Analysis by Negin
% =========================================================================

clearvars; close all; clc;

%% 1. PREPROCESSING (From MReza's Code)
fprintf('Loading data.mat and NotchFilter.mat...\n');
load('data.mat', 'dati');   
Fs = double(dati.fs);               
EEG = double(dati.eeg(:)); 

load('NotchFilter.mat', 'SOS', 'G');
EEG_raw = EEG;

% Detrend to remove DC offset
EEG = detrend(EEG);

% Apply Notch Filter (50 Hz) 
EEG = filtfilt(SOS, G, EEG);

% Bandpass Filter (0.5 to 30 Hz)
bp_filter = designfilt('bandpassiir', 'FilterOrder', 4, ...
    'HalfPowerFrequency1', 0.5, 'HalfPowerFrequency2', 30, ...
    'DesignMethod', 'butter', 'SampleRate', Fs);
EEG_filtered = filtfilt(bp_filter, EEG);

%% 2. INITIALIZATION FOR EPOCHING
epoch_sec = 30;
window_length = round(Fs * epoch_sec);
nEpochs = floor(length(EEG_filtered) / window_length);

% --- Negin's Variables (Welch PSD & Temporal Stats) ---
bandNames = {'Delta','Theta','Alpha','Sigma','Beta'}; 
bands = [ 0.5 4; 4 8; 8 12; 12 15; 15 30];
nBands = size(bands, 1);

RMS_val = zeros(nEpochs, 1); 
SignalStd = zeros(nEpochs, 1); 
TotalPower = zeros(nEpochs, 1);
RelativeBandPower = zeros(nEpochs, nBands);

welchWindow = hamming(round(Fs)); 
welchOverlap = floor(length(welchWindow)/2); 
nfftWelch = 2048;

% --- MReza's Variables (Features for Hypnogram & Plotting) ---
% STFT parameters (Defined globally to prevent plotting errors)
segment_length = round(Fs * 1);    
noverlap_stft  = round(Fs * 0.5);        
nfft_stft      = 2048; 

STFT_SlowWave = zeros(1, nEpochs); % Internal rule for N3
STFT_Sigma    = zeros(1, nEpochs); % Internal rule for N2
STFT_Alpha    = zeros(1, nEpochs); % Internal rule for Wake
time_min      = zeros(1, nEpochs);

% Explicit variables for all 5 bands requested for the presentation plots
Plot_Delta = zeros(1, nEpochs);
Plot_Theta = zeros(1, nEpochs);
Plot_Alpha = zeros(1, nEpochs);
Plot_Sigma = zeros(1, nEpochs);
Plot_Beta  = zeros(1, nEpochs);

fprintf('Processing %d epochs for PSD and Hypnogram features...\n', nEpochs);

%% 3. MAIN LOOP: NEGIN'S PSD + MREZA'S HYPNOGRAM FEATURES
for k = 1:nEpochs
    start_idx = (k - 1) * window_length + 1;
    end_idx   = k * window_length;
    epoch_data = detrend(EEG_filtered(start_idx:end_idx));
    time_min(k) = (start_idx + window_length/2) / (Fs * 60);
    
    % ========================================================
    % A. Negin's Statistical & Welch PSD Analysis
    % ========================================================
    RMS_val(k) = sqrt(mean(epoch_data.^2));
    SignalStd(k) = std(epoch_data);
    
    [pxx, f_w] = pwelch(epoch_data, welchWindow, welchOverlap, nfftWelch, Fs);
    idxTotal = f_w >= 0.5 & f_w <= 30;
    TotalPower(k) = trapz(f_w(idxTotal), pxx(idxTotal));
    
    for j = 1:nBands 
        idx = f_w >= bands(j,1) & f_w < bands(j,2); 
        if nnz(idx) >= 2 
            band_pow = trapz(f_w(idx), pxx(idx));
            RelativeBandPower(k, j) = band_pow / TotalPower(k);
        end 
    end
    
    % ========================================================
    % B. MReza's STFT Features (For AASM Stage Detection & Plots)
    % ========================================================
    [PSD_stft, f_s] = spectrogram(epoch_data, hamming(segment_length), noverlap_stft, nfft_stft, Fs, 'yaxis');
    PSD_stft = abs(PSD_stft).^2;
    Norm_PSD = bsxfun(@rdivide, PSD_stft, sum(PSD_stft, 1) + eps);
    
    % Extract Band Powers for decision logic (Huang 2014)
    sw_pow    = sum(Norm_PSD((f_s>=0.5 & f_s<=2), :), 1);
    alpha_pow = sum(Norm_PSD((f_s>=8   & f_s<=12), :), 1);
    sigma_pow = sum(Norm_PSD((f_s>=12  & f_s<=14), :), 1);
    
    % MReza's percentiles based on Huang 2014
    sw_sorted = sort(sw_pow, 'ascend');
    STFT_SlowWave(k) = mean(sw_sorted(end - floor(0.8*59) + 1 : end)); % Upper 80%
    
    alpha_sorted = sort(alpha_pow, 'ascend');
    STFT_Alpha(k) = mean(alpha_sorted(end - floor(0.5*59) + 1 : end)); % Upper 50%
    
    STFT_Sigma(k) = max(sigma_pow);
    
    % Extract all 5 explicit bands for the presentation figures (Mean of epoch)
    Plot_Delta(k) = mean(sum(Norm_PSD((f_s>=0.5 & f_s<=4), :), 1));
    Plot_Theta(k) = mean(sum(Norm_PSD((f_s>=4   & f_s<=8), :), 1));
    Plot_Alpha(k) = mean(sum(Norm_PSD((f_s>=8   & f_s<=12), :), 1));
    Plot_Sigma(k) = mean(sum(Norm_PSD((f_s>=12  & f_s<=15), :), 1));
    Plot_Beta(k)  = mean(sum(Norm_PSD((f_s>=15  & f_s<=30), :), 1));
end

%% 4. KNOWLEDGE-BASED STAGE DETECTION (MReza's Logic)
fprintf('Detecting sleep stages...\n');
Detected_Stage = strings(1, nEpochs);

thresh_alpha = mean(STFT_Alpha) + std(STFT_Alpha);
thresh_sw    = mean(STFT_SlowWave) + 0.5 * std(STFT_SlowWave);
thresh_sigma = mean(STFT_Sigma) + std(STFT_Sigma);

for k = 1:nEpochs
    if STFT_Alpha(k) > thresh_alpha
        Detected_Stage(k) = "Wake";
    elseif STFT_SlowWave(k) > thresh_sw
        Detected_Stage(k) = "N3";
    elseif STFT_Sigma(k) > thresh_sigma
        Detected_Stage(k) = "N2";
    else
        Detected_Stage(k) = "N1 / REM";
    end
end

%% 5. GENERATE PRESENTATION FIGURES
fprintf('Saving figures for presentation...\n');

% --- Slide 1: Preprocessing (MReza) ---
fig1 = figure('Name', 'Preprocessing', 'Position', [100 100 800 500]);
t_30s = (0:window_length-1) / Fs;
subplot(2,1,1); plot(t_30s, EEG_raw(1:window_length), 'Color', [0.6 0.6 0.6]);
title('Raw EEG Signal'); xlabel('Time [s]'); ylabel('Amplitude'); grid on;
subplot(2,1,2); plot(t_30s, EEG_filtered(1:window_length), 'b');
title('Filtered EEG (Detrend + Notch + Bandpass)'); xlabel('Time [s]'); ylabel('Amplitude'); grid on;
saveas(fig1, 'Slide1_Preprocessing.png');

% --- Slide 2: Global Welch PSD (Negin) ---
fig2 = figure('Name', 'Global Welch PSD', 'Position', [150 150 800 450]);
[pxx_glob, f_glob] = pwelch(EEG_filtered, welchWindow, welchOverlap, nfftWelch, Fs);
plot(f_glob, 10*log10(pxx_glob+eps), 'c', 'LineWidth', 2);
xlim([0 35]); 
title('Global Welch Power Spectral Density (PSD)');
xlabel('Frequency (Hz)'); ylabel('PSD (dB/Hz)'); grid on;
xline(4, '--b', 'Delta'); 
xline(8, '--g', 'Theta'); 
xline(12, '--r', 'Alpha'); 
xline(15, '--m', 'Sigma'); 
xline(30, '--y', 'Beta', 'LabelHorizontalAlignment', 'left'); 
saveas(fig2, 'Slide2_Global_PSD.png');

% --- Slide 3: ALL 5 STFT FEATURE BANDS ---
fig3 = figure('Name', 'STFT 5-Band Features', 'Position', [200 100 1000 850]);

ax(1) = subplot(6,1,1);
% Extract variables instead of auto-plotting to manually set time to minutes
[~, F_spec, T_spec, P_spec] = spectrogram(EEG_filtered, hamming(segment_length), noverlap_stft, nfft_stft, Fs);
imagesc(T_spec / 60, F_spec, 10*log10(P_spec + eps));
axis xy; % Ensure frequency 0 is at the bottom
ylim([0 35]); 
title('STFT Spectrogram');
ylabel('Frequency (Hz)');
cb = colorbar; cb.Label.String = 'Power/frequency (dB/Hz)';

ax(2) = subplot(6,1,2); plot(time_min, Plot_Delta, 'b', 'LineWidth', 1.5); 
title('Delta Band (0.5 - 4 Hz)'); ylabel('Power'); grid on;

ax(3) = subplot(6,1,3); plot(time_min, Plot_Theta, 'g', 'LineWidth', 1.5); 
title('Theta Band (4 - 8 Hz)'); ylabel('Power'); grid on;

ax(4) = subplot(6,1,4); plot(time_min, Plot_Alpha, 'r', 'LineWidth', 1.5); 
title('Alpha Band (8 - 12 Hz)'); ylabel('Power'); grid on;

ax(5) = subplot(6,1,5); plot(time_min, Plot_Sigma, 'm', 'LineWidth', 1.5); 
title('Sigma Band (12 - 15 Hz)'); ylabel('Power'); grid on;

ax(6) = subplot(6,1,6); plot(time_min, Plot_Beta, 'y', 'LineWidth', 1.5); 
title('Beta Band (15 - 30 Hz)'); ylabel('Power'); xlabel('Time (min)'); grid on;

% Link x-axes across all subplots
linkaxes(ax, 'x');

% Apply exact time limit to X-axis to eliminate the black void in the spectrogram
xlim(ax(1), [0, time_min(end)]); 

saveas(fig3, 'Slide3_STFT_All_Bands.png');

% --- Slide 4: Statistical Correlation Matrix (Negin) ---
fig4 = figure('Name', 'Correlation Matrix', 'Position', [250 250 600 500]);
corrMatrix = corr(RelativeBandPower, 'Rows', 'pairwise');
imagesc(corrMatrix); colorbar; clim([-1 1]); colormap('parula');
xticks(1:nBands); yticks(1:nBands);
xticklabels(bandNames); yticklabels(bandNames);
title('Correlation of Relative Band Powers');
for r = 1:nBands 
    for c = 1:nBands 
        text(c, r, sprintf('%.2f', corrMatrix(r,c)), 'HorizontalAlignment', 'center', 'Color', 'w', 'FontWeight', 'bold');
    end 
end
saveas(fig4, 'Slide4_Correlation_Matrix.png');

% --- Slide 5: Hypnogram Output (MReza) ---
fig5 = figure('Name', 'Hypnogram', 'Position', [300 300 900 400]);
stage_num = zeros(1, nEpochs);
stage_num(Detected_Stage == "Wake") = 0;
stage_num(Detected_Stage == "N1 / REM") = -1;
stage_num(Detected_Stage == "N2") = -2;
stage_num(Detected_Stage == "N3") = -3;
stairs(time_min, stage_num, 'LineWidth', 2, 'Color', 'b');
yticks([-3, -2, -1, 0]); yticklabels({'N3 (Deep)', 'N2 (Light)', 'N1 / REM', 'Wake'});
ylim([-3.5 0.5]); xlabel('Time [minutes]'); ylabel('Sleep Stage');
title('Automated Hypnogram (AASM Rules)'); grid on;
saveas(fig5, 'Slide5_Hypnogram.png');

fprintf('All 5 slides generated successfully!\n');