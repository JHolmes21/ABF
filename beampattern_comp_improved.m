
% Beampattern Comparison from Scratch: MVDR, SMI (clean), DL-SMI, LMS
clear; clc;

%% Array setup
N = 10;                      % Number of array elements
d = 0.5;                     % Spacing (wavelengths)
theta_scan = -90:0.1:90;     % Scan angles
snapshots = 1000;            % Number of snapshots
theta_desired = 10;          % Desired signal direction
theta_jammer = 45;           % Jammer direction
SNR_dB = 20;                 % Signal-to-noise ratio
INR_dB = 30;                 % Interference-to-noise ratio
alpha = 0.1;                 % Diagonal loading
mu = 0.01;                   % LMS step size

a = @(theta) exp(1j*2*pi*d*(0:N-1)'*sind(theta));  % Steering vector

%% Generate signals
s = sqrt(10^(SNR_dB/10)) * (randn(1, snapshots) + 1j * randn(1, snapshots)) / sqrt(2);
j = sqrt(10^(INR_dB/10)) * (randn(1, snapshots) + 1j * randn(1, snapshots)) / sqrt(2);
noise = (randn(N, snapshots) + 1j * randn(N, snapshots)) / sqrt(2);

a_s = a(theta_desired);
a_j = a(theta_jammer);

% Full received signal (desired + jammer + noise)
X_full = a_s * s + a_j * j + noise;

% Interference + noise only (for SMI)
X_jn = a_j * j + noise;

%% MVDR with full covariance
R_mvdr = (X_full * X_full') / snapshots;
w_mvdr = (R_mvdr \ a_s) / (a_s' * (R_mvdr \ a_s));

%% SMI (using only jammer + noise)
R_smi = (X_jn * X_jn') / snapshots;
w_smi = (R_smi \ a_s) / (a_s' * (R_smi \ a_s));

%% DL-SMI
R_dl = R_smi + alpha * eye(N);
w_dl = (R_dl \ a_s) / (a_s' * (R_dl \ a_s));

%% LMS adaptation
w_lms = zeros(N,1);
for n = 1:snapshots
    x_n = X_full(:,n);
    e = s(n) - w_lms' * x_n;
    w_lms = w_lms + mu * x_n * conj(e);
end

%% Beampattern computation
P_mvdr = zeros(size(theta_scan));
P_smi  = zeros(size(theta_scan));
P_dl   = zeros(size(theta_scan));
P_lms  = zeros(size(theta_scan));

for i = 1:length(theta_scan)
    a_theta = a(theta_scan(i));
    P_mvdr(i) = abs(w_mvdr' * a_theta).^2;
    P_smi(i)  = abs(w_smi'  * a_theta).^2;
    P_dl(i)   = abs(w_dl'   * a_theta).^2;
    P_lms(i)  = abs(w_lms'  * a_theta).^2;
end

% Normalize and convert to dB
P_mvdr = 10*log10(P_mvdr / max(P_mvdr));
P_smi  = 10*log10(P_smi  / max(P_mvdr));
P_dl   = 10*log10(P_dl   / max(P_mvdr));
P_lms  = 10*log10(P_lms  / max(P_mvdr));

%% Plot results
figure;
plot(theta_scan, P_mvdr, 'b-', 'LineWidth', 1.5); hold on;
plot(theta_scan, P_smi,  'r--', 'LineWidth', 1.5);
plot(theta_scan, P_dl,   'k-.', 'LineWidth', 1.5);
plot(theta_scan, P_lms,  'g:', 'LineWidth', 1.5);
xlabel('Angle (degrees)');
ylabel('Normalized Power (dB)');
title('Beampattern Comparison (Fully Rewritten)');
legend('MVDR', 'SMI (clean)', 'SMI + DL', 'LMS');
grid on;
axis([-90 90 -60 0]);
