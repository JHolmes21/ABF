% compare_mvdr_vs_smi.m
clear; clc; close all;

% Parameters
M = 10;                    % Number of array elements
d = 0.5;                   % Element spacing (wavelengths)
theta_desired = 30;        % Desired signal direction in degrees
theta_interf = [60, -40];  % Interferers
SNR_dB = 20;               % Signal-to-noise ratio
INR_dB = 30;               % Interference-to-noise ratio
N_snapshots = 200;         % Number of snapshots
theta_scan = -90:0.5:90;   % Scan angles

function BP = compute_beampattern(w, theta_scan, steering_func)
    BP = zeros(size(theta_scan));
    for k = 1:length(theta_scan)
        a_theta = steering_func(theta_scan(k));
        BP(k) = abs(w' * a_theta);
    end
    BP = BP / max(BP);  % Normalize
end


% Generate steering vectors
a = @(theta) exp(1j*2*pi*d*(0:M-1)'*sin(deg2rad(theta)));

% True covariance matrix (ideal)
a_des = a(theta_desired);
R_true = (10^(SNR_dB/10)) * (a_des * a_des');
for k = 1:length(theta_interf)
    a_int = a(theta_interf(k));
    R_true = R_true + (10^(INR_dB/10)) * (a_int * a_int');
end
R_true = R_true + eye(M); % Add noise

% Ideal MVDR weights
w_ideal = (R_true \ a_des) / (a_des' * (R_true \ a_des));

% Simulate received data
s = sqrt(10^(SNR_dB/10)) * randn(1, N_snapshots);       % Desired
X = a_des * s;
for k = 1:length(theta_interf)
    i = sqrt(10^(INR_dB/10)) * randn(1, N_snapshots);
    X = X + a(theta_interf(k)) * i;
end
X = X + randn(M, N_snapshots);                          % Add noise

% Sample covariance matrix (SMI)
R_smi = (X * X') / N_snapshots;

% MVDR weights using SMI
w_smi = (R_smi \ a_des) / (a_des' * (R_smi \ a_des));

% Compute beampatterns
BP_ideal = compute_beampattern(w_ideal, theta_scan, a);
BP_smi = compute_beampattern(w_smi, theta_scan, a);

% Plot
figure;
plot(theta_scan, 20*log10(BP_ideal), 'LineWidth', 1.5); hold on;
plot(theta_scan, 20*log10(BP_smi), '--r', 'LineWidth', 1.5);
legend('Ideal MVDR', 'SMI MVDR');
xlabel('Angle (degrees)');
ylabel('Beampattern (dB)');
title('MVDR vs SMI Beamformer Comparison');
grid on;
ylim([-60 0]);

