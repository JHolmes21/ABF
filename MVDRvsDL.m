% compare_mvdr_vs_smi_loading.m
clear; clc; close all;

% Parameters
M = 10;                    % Number of array elements
d = 0.5;                   % Element spacing (wavelengths)
theta_desired = 30;        % Desired signal direction in degrees
theta_interf = [60, -40];  % Interferers
SNR_dB = 20;               % Signal-to-noise ratio
INR_dB = 30;               % Interference-to-noise ratio
N_snapshots = 50;          % Small number of snapshots to show benefit of loading
theta_scan = -90:0.5:90;   % Scan angles
loading_factor = 1e-1;     % Diagonal loading factor (relative to identity)

% Steering vector function
a = @(theta) exp(1j*2*pi*d*(0:M-1)'*sin(deg2rad(theta)));

% Generate true covariance matrix
a_des = a(theta_desired);
R_true = (10^(SNR_dB/10)) * (a_des * a_des');
for k = 1:length(theta_interf)
    a_int = a(theta_interf(k));
    R_true = R_true + (10^(INR_dB/10)) * (a_int * a_int');
end
R_true = R_true + eye(M); % Add white noise

% Ideal MVDR weights
w_ideal = (R_true \ a_des) / (a_des' * (R_true \ a_des));

% Simulate data
s = sqrt(10^(SNR_dB/10)) * randn(1, N_snapshots);
X = a_des * s;
for k = 1:length(theta_interf)
    i = sqrt(10^(INR_dB/10)) * randn(1, N_snapshots);
    X = X + a(theta_interf(k)) * i;
end
X = X + randn(M, N_snapshots);  % Add noise

% SMI covariance
R_smi = (X * X') / N_snapshots;

% MVDR weights without loading
w_smi = (R_smi \ a_des) / (a_des' * (R_smi \ a_des));

% MVDR weights with diagonal loading
R_loaded = R_smi + loading_factor * eye(M);
w_loaded = (R_loaded \ a_des) / (a_des' * (R_loaded \ a_des));

% Compute beampatterns
BP_ideal  = compute_beampattern(w_ideal, theta_scan, a);
BP_smi    = compute_beampattern(w_smi, theta_scan, a);
BP_loaded = compute_beampattern(w_loaded, theta_scan, a);

% Plot
figure;
plot(theta_scan, 20*log10(BP_ideal), 'k', 'LineWidth', 1.5); hold on;
plot(theta_scan, 20*log10(BP_smi), '--r', 'LineWidth', 1.5);
plot(theta_scan, 20*log10(BP_loaded), '-.b', 'LineWidth', 1.5);
legend('Ideal MVDR', 'SMI MVDR', 'SMI + Loading');
xlabel('Angle (degrees)');
ylabel('Beampattern (dB)');
title('MVDR vs SMI with Diagonal Loading');
grid on;
ylim([-60 0]);

function BP = compute_beampattern(w, theta_scan, steering_func)
    BP = zeros(size(theta_scan));
    for k = 1:length(theta_scan)
        a_theta = steering_func(theta_scan(k));
        BP(k) = abs(w' * a_theta);
    end
    BP = BP / max(BP);  % Normalize
end
