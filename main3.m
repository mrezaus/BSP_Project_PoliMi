% =========================================================================
% SLEEP STAGE CLASSIFICATION (FEATURE BASED METHOD)
% Based on Huang et al. (2014) & Dement et al. (1957)
% Unified and Optimized Version 
% =========================================================================

clearvars; close all; clc;

%% 1. Load and Preprocess Data
fprintf('Loading data.mat and NotchFilter.mat...\n');
load('data.mat', 'dati');   
Fs = dati.fs;               
EEG = double(dati.eeg(:)); 

% Load the Notch filter provided by the professor
load('NotchFilter.mat', 'SOS', 'G');

% Store a copy of the raw signal for plotting
EEG_raw = EEG;

% Detrend the signal to remove DC offset and slow drifts
EEG = detrend(EEG);

% --- Apply Notch Filter (50 Hz) ---
% Applied explicitly as provided in the course materials
EEG = filtfilt(SOS, G, EEG);

% --- Design and Apply Bandpass Filter (0.5 to 30 Hz) ---
low_cutoff = 0.5;
high_cutoff = 30;
bp_filter = designfilt('bandpassiir', 'FilterOrder', 4, ...
    'HalfPowerFrequency1', low_cutoff, 'HalfPowerFrequency2', high_cutoff, ...
    'DesignMethod', 'butter', 'SampleRate', Fs);
EEG = filtfilt(bp_filter, EEG);

%% 2. Feature Extraction (Strictly based on Huang et al., 2014)
% Standard AASM epoch size
window_length = Fs * 30;              
nEpochs = floor(length(EEG) / window_length); % Exact number of full 30s epochs

% STFT parameters defined in the paper: 
% 1-second Hamming window overlapped with 0.5 seconds
segment_length = Fs * 1;    
noverlap = Fs * 0.5;        
nfft = 2048;                

% Pre-allocate feature arrays for speed
SlowWave_Power = zeros(1, nEpochs);
Theta_Power    = zeros(1, nEpochs);
Alpha_Power    = zeros(1, nEpochs);
Sigma_Power    = zeros(1, nEpochs);
time_min       = zeros(1, nEpochs);

for k = 1:nEpochs
    % Extract current 30-second epoch
    start_idx = (k - 1) * window_length + 1;
    end_idx   = k * window_length;
    EEG_epoch = detrend(EEG(start_idx:end_idx));
    
    % Compute Short-Time Fourier Transform (STFT) 
    % Generates 59 segments per epoch as described in Huang 2014
    [PSD_matrix, f] = spectrogram(EEG_epoch, hamming(segment_length), noverlap, nfft, Fs, 'yaxis');
    
    % Convert to absolute Power Spectral Density (PSD)
    PSD_matrix = abs(PSD_matrix).^2;
    
    % Normalization: Divide each frequency bin by the total sum in its segment
    total_power_per_segment = sum(PSD_matrix, 1);
    % Prevent division by zero
    total_power_per_segment(total_power_per_segment == 0) = eps; 
    Normalized_PSD = bsxfun(@rdivide, PSD_matrix, total_power_per_segment);
    
    % Exact frequency bands according to Huang et al., 2014
    sw_idx    = (f >= 0.5 & f <= 2);  % Slow Wave Activity for N3
    theta_idx = (f >= 5   & f <= 7);  % Theta Rhythm for N1/N2 background
    alpha_idx = (f >= 8   & f <= 12); % Alpha Rhythm for Wake
    sigma_idx = (f >= 12  & f <= 14); % Sleep Spindle for N2
    
    % Calculate relative power in specific bands across the 59 segments
    sw_pow    = sum(Normalized_PSD(sw_idx, :), 1);
    theta_pow = sum(Normalized_PSD(theta_idx, :), 1);
    alpha_pow = sum(Normalized_PSD(alpha_idx, :), 1);
    sigma_pow = sum(Normalized_PSD(sigma_idx, :), 1);
    
    % --- Epoch Features: Applying exact manual scoring properties ---
    % N3 (Slow Wave): Mean of the upper 80% of the 59 segments
    sw_sorted = sort(sw_pow, 'ascend');
    n80 = max(1, floor(0.8 * length(sw_sorted)));
    SlowWave_Power(k) = mean(sw_sorted(end-n80+1:end));
    
    % Wake (Alpha): Mean of the upper 50% of the 59 segments
    alpha_sorted = sort(alpha_pow, 'ascend');
    n50 = max(1, floor(0.5 * length(alpha_sorted)));
    Alpha_Power(k) = mean(alpha_sorted(end-n50+1:end));
    
    % N2 (Sigma/Spindles): Max power among the 59 segments (due to sudden short bursts)
    Sigma_Power(k) = max(sigma_pow);
    
    % N1/REM (Theta): Mean value
    Theta_Power(k) = mean(theta_pow);
    
    % Time vector center in minutes
    time_min(k) = (start_idx + window_length/2) / (Fs * 60);
end

%% 3. Threshold-Based Stage Detection (Translating AASM Rules)
Detected_Stage = strings(1, nEpochs);

% Adaptive thresholds based on subject's overall physiological baseline
% Note: Using robust statistics (median and absolute deviation can also be used)
thresh_alpha = mean(Alpha_Power) + std(Alpha_Power);
thresh_sw    = mean(SlowWave_Power) + 0.5 * std(SlowWave_Power);
thresh_sigma = mean(Sigma_Power) + std(Sigma_Power);

for k = 1:nEpochs
    % AASM Rule: Dominant alpha rhythm -> Wake
    if Alpha_Power(k) > thresh_alpha
        Detected_Stage(k) = "Wake";
        
    % AASM Rule: High amplitude slow wave activity -> N3
    elseif SlowWave_Power(k) > thresh_sw
        Detected_Stage(k) = "N3";
        
    % AASM Rule: Sleep spindle occurrence (max sigma burst) -> N2
    elseif Sigma_Power(k) > thresh_sigma
        Detected_Stage(k) = "N2";
        
    % AASM Rule: Low voltage mixed frequency (Alpha attenuated, Theta dominant) -> N1/REM
    % Note: distinguishing N1 from REM accurately requires EOG/EMG channels.
    else
        Detected_Stage(k) = "N1 / REM";
    end
end

%% 4. Generate and Save Presentation Figures
fprintf('Generating and saving figures...\n');

% --- Figure 1: Raw vs Filtered Signal ---
fig1 = figure('Name', 'Preprocessing', 'NumberTitle', 'off', 'Position', [100 100 800 600]);
t_30s = (0:window_length-1) / Fs;
subplot(2,1,1); plot(t_30s, EEG_raw(1:window_length), 'Color', [0.6 0.6 0.6]);
title('Raw EEG Signal (First 30 Seconds Epoch)'); ylabel('Amplitude'); xlabel('Time [s]'); grid on;
subplot(2,1,2); plot(t_30s, EEG(1:window_length), 'Color', 'b');
title('Filtered EEG Signal (Detrend + Notch + Bandpass)'); ylabel('Amplitude'); xlabel('Time [s]'); grid on;
saveas(fig1, 'FigX_1_1.png');

% --- Figure 2: STFT Spectrogram ---
fig2 = figure('Name', 'Spectrogram', 'NumberTitle', 'off', 'Position', [150 150 800 500]);
spectrogram(EEG, hamming(segment_length), noverlap, nfft, Fs, 'yaxis');
ylim([0 30]); title('STFT Spectrogram (Frequency Content Over Time)'); colormap jet;
saveas(fig2, 'FigX_1_2.png');

% --- Figure 3: Extracted Features Based on Huang et al. 2014 ---
fig3 = figure('Name', 'Extracted Features', 'NumberTitle', 'off', 'Position', [200 200 900 700]);
ax(1) = subplot(4,1,1); plot(time_min, SlowWave_Power, 'LineWidth', 1.5); ylabel('Slow Wave'); title('N3 Indicator (Upper 80% Mean)'); grid on;
ax(2) = subplot(4,1,2); plot(time_min, Alpha_Power, 'LineWidth', 1.5); ylabel('Alpha'); title('Wake Indicator (Upper 50% Mean)'); grid on;
ax(3) = subplot(4,1,3); plot(time_min, Sigma_Power, 'LineWidth', 1.5); ylabel('Sigma'); title('N2 Indicator (Maximum Burst)'); grid on;
ax(4) = subplot(4,1,4); plot(time_min, Theta_Power, 'LineWidth', 1.5, 'Color', 'r'); xlabel('Time [minutes]'); ylabel('Theta'); title('N1 Background (Mean Theta)'); grid on;
linkaxes(ax, 'x');
saveas(fig3, 'FigX_1_3.png');

% --- Figure 4: Hypnogram (Output) ---
fig4 = figure('Name', 'Hypnogram', 'NumberTitle', 'off', 'Position', [250 250 900 400]);
stage_num = zeros(1, nEpochs);
stage_num(Detected_Stage == "Wake") = 0;
stage_num(Detected_Stage == "N1 / REM") = -1;
stage_num(Detected_Stage == "N2") = -2;
stage_num(Detected_Stage == "N3") = -3;

% Plot Hypnogram in Blue
stairs(time_min, stage_num, 'LineWidth', 2, 'Color', 'b');
yticks([-3, -2, -1, 0]); yticklabels({'N3 (Deep)', 'N2 (Light)', 'N1 / REM', 'Wake'});
ylim([-3.5 0.5]); xlabel('Time [minutes]'); ylabel('Sleep Stage');
title('Predicted Hypnogram Based on AASM Rules & Huang (2014) Features'); grid on;
saveas(fig4, 'FigX_1_4.png');

fprintf('All figures have been saved successfully!\n');