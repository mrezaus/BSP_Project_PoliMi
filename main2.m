% =========================================================================
% SLEEP STAGE CLASSIFICATION (UNSUPERVISED EDITION)
% Based on Huang et al. (2014) & Dement et al. (1957)
% Author : Mohammadreza Iravani
% =========================================================================

% 1. Load and Extract Data
fprintf('Loading data.mat...\n');
load('data.mat', 'dati');   
fs = dati.fs;               
raw_eeg = dati.eeg;         

% 2. Epoching (30-second windows)
epoch_len = 30 * fs;        
num_epochs = floor(length(raw_eeg) / epoch_len);
eeg_epochs = reshape(raw_eeg(1:num_epochs*epoch_len), epoch_len, num_epochs)';

% 3. Feature Extraction (Based on Huang 2014 rules)
num_features = 13;
X = zeros(num_epochs, num_features);

fprintf('Extracting 13 frequency-domain features for %d epochs...\n', num_epochs);
for i = 1:num_epochs
    X(i, :) = extract_huang_single_channel(eeg_epochs(i, :), fs);
end

% 4. Standardization
mu = mean(X, 1);
sigma = std(X, 0, 1);
sigma(sigma == 0) = 1; 
X_scaled = (X - mu) ./ sigma;

% 5. Unsupervised Classification (K-Means Clustering)
fprintf('Performing K-Means clustering (K=3: Wake, NREM, REM)...\n');
rng(42); % For reproducibility
num_clusters = 3;
[cluster_idx, cluster_centers] = kmeans(X_scaled, num_clusters, 'Replicates', 10);

% 6. Knowledge-Based Centroid Mapping
% Feature Index mapping from extract_huang_single_channel:
% 1,2: Lower Delta | 3,4: Delta | 5,6: Theta | 7,8: Alpha | 9,10: Sigma | 11,12: Beta | 13: Gamma
delta_power = mean(cluster_centers(:, 1:4), 2);
alpha_beta_power = mean(cluster_centers(:, [7, 8, 11, 12, 13]), 2);

% Identify Wake cluster: Highest Alpha/Beta/Gamma
[~, wake_cluster] = max(alpha_beta_power);

% Out of the remaining two, identify NREM (Highest Delta) and REM
remaining_clusters = setdiff(1:3, wake_cluster);
if delta_power(remaining_clusters(1)) > delta_power(remaining_clusters(2))
    nrem_cluster = remaining_clusters(1);
    rem_cluster = remaining_clusters(2);
else
    nrem_cluster = remaining_clusters(2);
    rem_cluster = remaining_clusters(1);
end

% Map clusters to numeric stages for plotting (1: NREM, 2: REM, 3: W)
numeric_stages = zeros(num_epochs, 1);
predicted_labels = cell(num_epochs, 1);
for i = 1:num_epochs
    if cluster_idx(i) == wake_cluster
        numeric_stages(i) = 3;
        predicted_labels{i} = 'W';
    elseif cluster_idx(i) == rem_cluster
        numeric_stages(i) = 2;
        predicted_labels{i} = 'REM';
    else
        numeric_stages(i) = 1;
        predicted_labels{i} = 'NREM';
    end
end

% =========================================================================
% PRESENTATION VISUALIZATIONS
% =========================================================================
set(0, 'DefaultAxesFontSize', 12, 'DefaultLineLineWidth', 1.5);
time_hrs = (1:num_epochs) * 30 / 3600; 

% FIGURE 1: Raw vs Preprocessed Signal
fig1 = figure('Name', 'Fig 1: Signal Preprocessing', 'Position', [50, 100, 1000, 500]);
sample_epoch = eeg_epochs(40, :); 
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

% FIGURE 2: Clinical Rhythm Decomposition
fig2 = figure('Name', 'Fig 2: Rhythm Decomposition', 'Position', [100, 100, 1000, 800]);
bands = {[1 4], [5 7], [8 12], [12 14], [15 30]};
band_names = {'Delta (1-4 Hz) - Slow Waves', 'Theta (5-7 Hz) - REM/Light Sleep', ...
              'Alpha (8-12 Hz) - Wakefulness', 'Sigma (12-14 Hz) - Spindles', ...
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
sgtitle('EEG Epoch Decomposition into Frequency Bands');

% FIGURE 3: Short-Time Fourier Transform (STFT)
fig3 = figure('Name', 'Fig 3: STFT', 'Position', [150, 150, 800, 400]);
window = hamming(fs); noverlap = fs / 2; nfft = fs;
[S, F, T, P] = spectrogram(filtered_epoch, window, noverlap, nfft, fs);
surf(T, F, 10*log10(P), 'EdgeColor', 'none');
axis xy; axis tight; colormap(jet); view(0,90);
c = colorbar; c.Label.String = 'Power (dB/Hz)';
title('STFT: Power Spectral Density');
xlabel('Time (s)'); ylabel('Frequency (Hz)'); ylim([0 40]); 

% FIGURE 4: Feature Space Separability (PCA & K-Means Clusters)
fig4 = figure('Name', 'Fig 4: PCA Clustering', 'Position', [200, 200, 800, 600]);
[~, score, ~] = pca(X_scaled);
gscatter(score(:,1), score(:,2), predicted_labels, 'rgb', '...', 20);
title('PCA: Unsupervised Sleep Stage Clustering (K-Means)');
xlabel('Principal Component 1'); ylabel('Principal Component 2');
legend('Location', 'best'); grid on;

% FIGURE 5: Full Night Hypnogram
fig5 = figure('Name', 'Fig 5: Full Hypnogram', 'Position', [250, 250, 1000, 400]);
stairs(time_hrs, numeric_stages, 'LineWidth', 1.5, 'Color', [0.0 0.3 0.6]); 
set(gca, 'YTick', 1:3, 'YTickLabel', {'NREM', 'REM', 'Wake'});
ylim([0.5 3.5]); xlim([0 max(time_hrs)]);
title('Automated Hypnogram via Unsupervised Clustering');
xlabel('Time in Bed (Hours)'); grid on;

% FIGURE 6: Zoomed Hypnogram 
fig6 = figure('Name', 'Fig 6: Zoomed Hypnogram', 'Position', [300, 300, 800, 400]);
stairs(time_hrs, numeric_stages, 'LineWidth', 2.0, 'Color', [0.8500 0.3250 0.0980]);
set(gca, 'YTick', 1:3, 'YTickLabel', {'NREM', 'REM', 'Wake'});
ylim([0.5 3.5]); xlim([0 2.0]); 
title('Hypnogram Detail (0 to 120 Mins)');
xlabel('Time in Bed (Hours)'); grid on;

% =========================================================================
% FIGURE EXPORT TO PNG
% =========================================================================
fprintf('Saving high-resolution figures to current folder...\n');
pause(1); 
exportgraphics(fig1, 'Fig1_Preprocessing.png', 'Resolution', 300);
exportgraphics(fig2, 'Fig2_RhythmDecomposition.png', 'Resolution', 300);
exportgraphics(fig3, 'Fig3_STFT_Spectrogram.png', 'Resolution', 300);
exportgraphics(fig4, 'Fig4_PCA_Clustering.png', 'Resolution', 300);
exportgraphics(fig5, 'Fig5_FullHypnogram.png', 'Resolution', 300);
exportgraphics(fig6, 'Fig6_ZoomedHypnogram.png', 'Resolution', 300);
fprintf('All images saved successfully as PNGs!\n');

% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================
function features = extract_huang_single_channel(epoch_data, fs)
    [b, a] = butter(4, [0.5, 49] / (fs / 2), 'bandpass');
    filtered_sig = filtfilt(b, a, epoch_data);
    
    window = hamming(fs); noverlap = fs / 2; nfft = fs; 
    [S, F, ~] = spectrogram(filtered_sig, window, noverlap, nfft, fs);
    Sxx = abs(S).^2;
    Sxx_norm = Sxx ./ sum(Sxx, 1);
    
    b_ldelta = (F >= 0.5 & F <= 2);
    b_delta  = (F >= 1   & F <= 4);
    b_theta  = (F >= 5   & F <= 7);
    b_alpha  = (F >= 8   & F <= 12);
    b_sigma  = (F >= 12  & F <= 14);
    b_beta   = (F >= 15  & F <= 30);
    b_gamma  = (F >= 30  & F <= 49); 
    
    p_ldelta = sum(Sxx_norm(b_ldelta, :), 1);
    p_delta  = sum(Sxx_norm(b_delta, :), 1);
    p_theta  = sum(Sxx_norm(b_theta, :), 1);
    p_alpha  = sum(Sxx_norm(b_alpha, :), 1);
    p_sigma  = sum(Sxx_norm(b_sigma, :), 1);
    p_beta   = sum(Sxx_norm(b_beta, :), 1);
    p_gamma  = sum(Sxx_norm(b_gamma, :), 1);
    
    num_seg = 59;
    top_80 = floor(0.8 * num_seg); 
    top_50 = floor(0.5 * num_seg); 
    
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