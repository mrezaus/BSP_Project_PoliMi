% =========================================================================
% SLEEP STAGE CLASSIFICATION (FEATURE BASED METHOD)
% Based on Huang et al. (2014) & Dement et al. (1957)
% Author : Mohammadreza Iravani
% =========================================================================

clearvars; close all; clc;

%% 1. Load and Preprocess Data
fprintf('Loading data.mat...\n');
load('data.mat', 'dati');   
Fs = dati.fs;               
EEG = dati.eeg; 

% Store a copy of the raw signal for plotting
EEG_raw = EEG;

% Detrend the signal to remove DC offset
EEG = detrend(EEG);

% --- Design and Apply Bandpass Filter (0.5 to 30 Hz) ---
low_cutoff = 0.5;
high_cutoff = 30;
bp_filter = designfilt('bandpassiir', 'FilterOrder', 4, ...
    'HalfPowerFrequency1', low_cutoff, 'HalfPowerFrequency2', high_cutoff, ...
    'DesignMethod', 'butter', 'SampleRate', Fs);
EEG = filtfilt(bp_filter, EEG);

%% 2. Feature Extraction (Based on Huang et al., 2014 via STFT)
% Standard AASM epoch size
window_length = Fs * 30;              
shift = window_length; 

% STFT parameters defined in the paper: 
% 1-second Hamming window overlapped with 0.5 seconds
segment_length = Fs * 1;    
noverlap = Fs * 0.5;        
nfft = 2048;                

N = length(EEG);
i = 1;  
counter = 1;

while (i + window_length - 1) < N
    EEG_epoch = detrend(EEG(i:i+window_length-1));
    
    % Compute Short-Time Fourier Transform (STFT) for the 30s epoch
    % This generates 59 segments per epoch as described in the paper
    [PSD_matrix, f] = spectrogram(EEG_epoch, hamming(segment_length), noverlap, nfft, Fs, 'yaxis');
    
    % Convert to absolute Power Spectral Density (PSD)
    PSD_matrix = abs(PSD_matrix).^2;
    
    % Normalization: Divide each frequency bin by the total sum in its segment
    total_power_per_segment = sum(PSD_matrix, 1);
    Normalized_PSD = bsxfun(@rdivide, PSD_matrix, total_power_per_segment);
    
    % Exact frequency bands according to AASM and Huang et al., 2014
    sw_idx    = (f >= 0.5 & f <= 2);  % Slow Wave Activity for N3
    theta_idx = (f >= 5   & f <= 7);  % Theta Rhythm for N1/N2 background
    alpha_idx = (f >= 8   & f <= 12); % Alpha Rhythm for Wake
    sigma_idx = (f >= 12  & f <= 14); % Sleep Spindle for N2
    
    % Calculate relative power in specific bands across the 59 segments
    SW_Power_Segments    = sum(Normalized_PSD(sw_idx, :), 1);
    Theta_Power_Segments = sum(Normalized_PSD(theta_idx, :), 1);
    Alpha_Power_Segments = sum(Normalized_PSD(alpha_idx, :), 1);
    Sigma_Power_Segments = sum(Normalized_PSD(sigma_idx, :), 1);
    
    % Epoch Features: Capturing temporal properties of manual scoring rules
    SlowWave_Power(counter) = mean(SW_Power_Segments);
    Theta_Power(counter)    = mean(Theta_Power_Segments);
    Alpha_Power(counter)    = mean(Alpha_Power_Segments);
    % Since spindles occur suddenly (0.5 - 1.5s), maximum power among 59 segments is extracted
    Sigma_Power(counter)    = max(Sigma_Power_Segments); 
    
    time_min(counter) = (i + window_length/2) / (Fs * 60);
    i = i + shift;
    counter = counter + 1;
end

%% 3. Threshold-Based Stage Detection (Translating AASM Rules)
Detected_Stage = strings(1, length(time_min));

% Adaptive thresholds based on subject's overall physiological baseline
thresh_alpha = mean(Alpha_Power) + std(Alpha_Power);
thresh_sw    = mean(SlowWave_Power) + 0.5 * std(SlowWave_Power);
thresh_sigma = mean(Sigma_Power) + std(Sigma_Power);

for k = 1:length(time_min)
    % AASM Rule: >50% alpha rhythm -> Wake
    if Alpha_Power(k) > thresh_alpha
        Detected_Stage(k) = "Wake";
        
    % AASM Rule: >20% high amplitude slow wave -> N3
    elseif SlowWave_Power(k) > thresh_sw
        Detected_Stage(k) = "N3";
        
    % AASM Rule: Sleep spindle (12-14Hz) occurrence -> N2
    elseif Sigma_Power(k) > thresh_sigma
        Detected_Stage(k) = "N2";
        
    % AASM Rule: Alpha is attenuated and replaced by Theta -> N1 / REM
    else
        Detected_Stage(k) = "N1 / REM";
    end
end

%% 4. Generate and Save Presentation Figures
fprintf('Generating and saving figures...\n');

% --- Figure 1: Raw vs Filtered Signal ---
fig1 = figure('Name', 'Preprocessing', 'NumberTitle', 'off');
t_30s = (0:window_length-1) / Fs;
subplot(2,1,1); plot(t_30s, EEG_raw(1:window_length), 'Color', [0.6 0.6 0.6]);
title('Raw EEG Signal (First 30 Seconds Epoch)'); ylabel('Amplitude'); xlabel('Time [s]'); grid on;
subplot(2,1,2); plot(t_30s, EEG(1:window_length), 'Color', 'b');
title('Filtered EEG Signal (0.5 - 30 Hz Bandpass)'); ylabel('Amplitude'); xlabel('Time [s]'); grid on;
% Save Figure 1
saveas(fig1, 'FigX_1_1.png');

% --- Figure 2: STFT Spectrogram ---
fig2 = figure('Name', 'Spectrogram', 'NumberTitle', 'off');
spectrogram(EEG, hamming(segment_length), noverlap, nfft, Fs, 'yaxis');
ylim([0 30]); title('STFT Spectrogram (Frequency Content Over Time)'); colormap jet;
% Save Figure 2
saveas(fig2, 'FigX_1_2.png');

% --- Figure 3: Extracted Features Based on Huang et al. 2014 ---
fig3 = figure('Name', 'Extracted Features', 'NumberTitle', 'off');
ax(1) = subplot(4,1,1); plot(time_min, SlowWave_Power, 'LineWidth', 1.5); ylabel('Slow Wave'); title('N3 Indicator (0.5 - 2 Hz)'); grid on;
ax(2) = subplot(4,1,2); plot(time_min, Alpha_Power, 'LineWidth', 1.5); ylabel('Alpha'); title('Wake Indicator (8 - 12 Hz)'); grid on;
ax(3) = subplot(4,1,3); plot(time_min, Sigma_Power, 'LineWidth', 1.5); ylabel('Sigma'); title('N2 Indicator (12 - 14 Hz)'); grid on;
ax(4) = subplot(4,1,4); plot(time_min, Theta_Power, 'LineWidth', 1.5, 'Color', 'r'); xlabel('Time [minutes]'); ylabel('Theta'); title('N1 Background (5 - 7 Hz)'); grid on;
linkaxes(ax, 'x');
% Save Figure 3
saveas(fig3, 'FigX_1_3.png');

% --- Figure 4: Hypnogram (Output) ---
fig4 = figure('Name', 'Hypnogram', 'NumberTitle', 'off');
stage_num = zeros(1, length(Detected_Stage));
stage_num(Detected_Stage == "Wake") = 0;
stage_num(Detected_Stage == "N1 / REM") = -1;
stage_num(Detected_Stage == "N2") = -2;
stage_num(Detected_Stage == "N3") = -3;

% Plot Hypnogram in Blue ('b')
stairs(time_min, stage_num, 'LineWidth', 2, 'Color', 'b');
yticks([-3, -2, -1, 0]); yticklabels({'N3 (Deep)', 'N2 (Light)', 'N1 / REM', 'Wake'});
ylim([-3.5 0.5]); xlabel('Time [minutes]'); ylabel('Sleep Stage');
title('Predicted Hypnogram Based on AASM Rules'); grid on;
% Save Figure 4
saveas(fig4, 'FigX_1_4.png');

fprintf('All figures have been saved successfully!\n');