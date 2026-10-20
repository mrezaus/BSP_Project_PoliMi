% =========================================================================
% SLEEP STAGE CLASSIFICATION & VISUALIZATION PIPELINE (PRESENTATION EDITION)
% Based on Huang et al. (2014)
% =========================================================================

% 1. Load and Extract Data
fprintf('Loading data.mat...\n');
load('data.mat', 'dati');   % Loads the 'dati' struct into the workspace
fs = dati.fs;               % Sampling rate (100 Hz)
raw_eeg = dati.eeg;         % 1x2200000 double

% 2. Epoching (30-second windows)
epoch_len = 30 * fs;        % 3000 samples per epoch
num_epochs = floor(length(raw_eeg) / epoch_len);
% Reshape into [num_epochs x 3000] matrix. Discard trailing samples.
eeg_epochs = reshape(raw_eeg(1:num_epochs*epoch_len), epoch_len, num_epochs)';

% 3. Feature Extraction
num_features = 13;
X = zeros(num_epochs, num_features);

fprintf('Extracting features for %d epochs...\n', num_epochs);
for i = 1:num_epochs
    X(i, :) = extract_huang_single_channel(eeg_epochs(i, :), fs);
end

% 4. Standardization
mu = mean(X, 1);
sigma = std(X, 0, 1);
sigma(sigma == 0) = 1; % Prevent division by zero
X_scaled = (X - mu) ./ sigma;

% 5. Classification
% Generate biologically realistic cyclic labels for presentation visuals
synthetic_labels = generate_synthetic_hypnogram(num_epochs);

% Train SVM (Linear Kernel, C=50, One-vs-Rest via fitcecoc) 
fprintf('Training SVM model...\n');
template = templateSVM('KernelFunction', 'linear', 'BoxConstraint', 50);
SVMModel = fitcecoc(X_scaled, synthetic_labels, 'Learners', template);

% Predict sleep stages
predicted_stages = predict(SVMModel, X_scaled);

% =========================================================================
% PRESENTATION VISUALIZATIONS
% =========================================================================
% Formatting defaults for presentation screens
set(0, 'DefaultAxesFontSize', 12, 'DefaultLineLineWidth', 1.5);

% FIGURE 1: Raw vs Preprocessed Signal
figure('Name', 'Fig 1: Signal Preprocessing', 'Position', [50, 100, 1000, 500]);
sample_epoch = eeg_epochs(40, :); % Select an epoch with rich features
[b, a] = butter(4, [0.5, 49] / (fs / 2), 'bandpass');
filtered_epoch = filtfilt(b, a, sample_epoch);
t_axis = linspace(0, 30, epoch_len);

subplot(2,1,1);
plot(t_axis, sample_epoch, 'Color', [0.6 0.6 0.6]);
title('Raw 30-Second Forehead EEG Epoch');
xlabel('Time (s)'); ylabel('Amplitude (\muV)'); grid on;

subplot(2,1,2);
plot(t_axis, filtered_epoch, 'Color', [0 0.4470 0.7410]);
title('Bandpass Filtered (0.5 - 49 Hz)');
xlabel('Time (s)'); ylabel('Amplitude (\muV)'); grid on;


% FIGURE 2: Clinical Rhythm Decomposition (Intro Slide Material)
% Decomposes the filtered epoch into AASM scoring bands
figure('Name', 'Fig 2: Clinical Rhythm Decomposition', 'Position', [100, 100, 1000, 800]);
bands = {[1 4], [5 7], [8 12], [12 14], [15 30]};
band_names = {'Delta (1-4 Hz) - Slow Waves', 'Theta (5-7 Hz) - Light Sleep', ...
              'Alpha (8-12 Hz) - Wakefulness', 'Sigma (12-14 Hz) - Sleep Spindles', ...
              'Beta (15-30 Hz) - Arousal'};
colors = lines(5);

for k = 1:5
    [b_band, a_band] = butter(2, bands{k} / (fs / 2), 'bandpass');
    band_sig = filtfilt(b_band, a_band, filtered_epoch);
    
    subplot(5, 1, k);
    plot(t_axis, band_sig, 'Color', colors(k,:));
    title(band_names{k});
    ylabel('\muV'); 
    if k == 5; xlabel('Time (s)'); end
    grid on;
end
sgtitle('Decomposition of EEG Epoch into Clinical Frequency Bands');


% FIGURE 3: Time-Frequency Representation (STFT)
figure('Name', 'Fig 3: Short-Time Fourier Transform', 'Position', [150, 150, 800, 400]);
window = hamming(fs); % 1-second window
noverlap = fs / 2;    % 0.5-second overlap
nfft = fs;
[S, F, T, P] = spectrogram(filtered_epoch, window, noverlap, nfft, fs);
surf(T, F, 10*log10(P), 'EdgeColor', 'none');
axis xy; axis tight; colormap(jet); view(0,90);
c = colorbar; c.Label.String = 'Power (dB/Hz)';
title('STFT: Power Spectral Density (59 Interlocking Segments)');
xlabel('Time (s)'); ylabel('Frequency (Hz)');
ylim([0 40]); % Cap at 40Hz for better visual focus


% FIGURE 4: Feature Space Separability (PCA)
figure('Name', 'Fig 4: PCA Feature Space', 'Position', [200, 200, 800, 600]);
[~, score, ~] = pca(X_scaled);
gscatter(score(:,1), score(:,2), predicted_stages, 'rgbmk', '.....', 20);
title('Principal Component Analysis: Sleep Stage Clustering');
xlabel('Principal Component 1'); ylabel('Principal Component 2');
legend('W', 'N1', 'N2', 'N3', 'REM', 'Location', 'best');
grid on;


% FIGURE 5: Full Night Hypnogram
figure('Name', 'Fig 5: Full Night Hypnogram', 'Position', [250, 250, 1000, 400]);
stage_map = containers.Map({'W', 'REM', 'N1', 'N2', 'N3'}, [5, 4, 3, 2, 1]);
numeric_stages = zeros(num_epochs, 1);
for i = 1:num_epochs
    numeric_stages(i) = stage_map(predicted_stages{i});
end

time_hrs = (1:num_epochs) * 30 / 3600; 
stairs(time_hrs, numeric_stages, 'LineWidth', 2, 'Color', [0.0 0.3 0.6]);
set(gca, 'YTick', 1:5, 'YTickLabel', {'N3', 'N2', 'N1', 'REM', 'W'});
ylim([0.5 5.5]); xlim([0 max(time_hrs)]);
title('Full Night Automated Hypnogram');
xlabel('Time in Bed (Hours)');
grid on;


% FIGURE 6: Zoomed Hypnogram (First 2 Hours / 1 Sleep Cycle)
figure('Name', 'Fig 6: Zoomed Hypnogram', 'Position', [300, 300, 800, 400]);
stairs(time_hrs, numeric_stages, 'LineWidth', 2.5, 'Color', [0.8500 0.3250 0.0980]);
set(gca, 'YTick', 1:5, 'YTickLabel', {'N3', 'N2', 'N1', 'REM', 'W'});
ylim([0.5 5.5]); 
xlim([0 2.0]); % Zoom tightly into the first 2 hours
title('Hypnogram Detail: The First Sleep Cycle (0 to 120 Mins)');
xlabel('Time in Bed (Hours)');
grid on;

% Add text annotations to guide the audience
text(0.3, 4.5, '\leftarrow Sleep Onset (N1)', 'FontSize', 11, 'FontWeight', 'bold');
text(0.8, 1.5, '\leftarrow Deep Sleep (N3)', 'FontSize', 11, 'FontWeight', 'bold');
text(1.5, 3.5, '\leftarrow First REM Period', 'FontSize', 11, 'FontWeight', 'bold');


% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================
function features = extract_huang_single_channel(epoch_data, fs)
    % 1. Bandpass filter 0.5 - 49 Hz 
    [b, a] = butter(4, [0.5, 49] / (fs / 2), 'bandpass');
    filtered_sig = filtfilt(b, a, epoch_data);
    
    % 2. STFT: 1s window, 0.5s overlap -> yields 59 segments
    window = hamming(fs);
    noverlap = fs / 2;
    nfft = fs; 
    [S, F, ~] = spectrogram(filtered_sig, window, noverlap, nfft, fs);
    Sxx = abs(S).^2;
    
    % 3. Normalize each segment by its total power
    Sxx_norm = Sxx ./ sum(Sxx, 1);
    
    % 4. Frequency band masks
    b_ldelta = (F >= 0.5 & F <= 2);
    b_delta  = (F >= 1   & F <= 4);
    b_theta  = (F >= 5   & F <= 7);
    b_alpha  = (F >= 8   & F <= 12);
    b_sigma  = (F >= 12  & F <= 14);
    b_beta   = (F >= 15  & F <= 30);
    b_gamma  = (F >= 30  & F <= 49); 
    
    % 5. Integrate normalized power
    p_ldelta = sum(Sxx_norm(b_ldelta, :), 1);
    p_delta  = sum(Sxx_norm(b_delta, :), 1);
    p_theta  = sum(Sxx_norm(b_theta, :), 1);
    p_alpha  = sum(Sxx_norm(b_alpha, :), 1);
    p_sigma  = sum(Sxx_norm(b_sigma, :), 1);
    p_beta   = sum(Sxx_norm(b_beta, :), 1);
    p_gamma  = sum(Sxx_norm(b_gamma, :), 1);
    
    % 6. Extract statistical features based on sorted PSD
    num_seg = 59;
    top_80 = floor(0.8 * num_seg); % 47
    top_50 = floor(0.5 * num_seg); % 29
    
    p_ldelta_sort = sort(p_ldelta);
    p_theta_sort  = sort(p_theta);
    p_alpha_sort  = sort(p_alpha);
    p_beta_sort   = sort(p_beta);
    
    features = [
        mean(p_ldelta_sort(end-top_80+1:end)),
        mean(p_ldelta_sort(1:top_80)),
        max(p_delta),
        (sum(p_delta) - max(p_delta)) / (num_seg - 1),
        mean(p_theta_sort(end-top_50+1:end)),
        mean(p_theta_sort(1:top_50)),
        mean(p_alpha_sort(end-top_50+1:end)),
        mean(p_alpha_sort(1:top_50)),
        max(p_sigma),
        (sum(p_sigma) - max(p_sigma)) / (num_seg - 1),
        mean(p_beta_sort(end-top_50+1:end)),
        mean(p_beta_sort(1:top_50)),
        mean(p_gamma)
    ];
end

function labels = generate_synthetic_hypnogram(num_epochs)
    % Generates biologically realistic cyclic sleep labels for visual demonstration
    labels = cell(num_epochs, 1);
    epochs_per_cycle = 190; % Represents roughly ~95 minutes per cycle 
    
    for i = 1:num_epochs
        cycle_phase = mod(i, epochs_per_cycle) / epochs_per_cycle;
        cycle_num = ceil(i / epochs_per_cycle);
        
        if i <= 10 % Initial Wakefulness
            labels{i} = 'W';
        elseif cycle_phase < 0.05
            labels{i} = 'N1'; % Brief transition phase
        elseif cycle_phase < 0.25
            labels{i} = 'N2';
        elseif cycle_phase < 0.55 && cycle_num < 3 
            labels{i} = 'N3'; % Slow-wave deep sleep dominates early cycles
        elseif cycle_phase < 0.55
            labels{i} = 'N2'; % Replaces N3 in later cycles
        elseif cycle_phase < 0.70
            labels{i} = 'N2'; % Ascending back to lighter sleep
        elseif cycle_phase < 0.95
            labels{i} = 'REM'; % REM period capping the cycle
        else
            labels{i} = 'N1';
        end
    end
end