
% Clean Beampattern Comparison from Scratch: MVDR, SMI, DL-SMI, LMS
clear; clc;

% Parameters
N = 10;                      % Number of array elements
d = 0.5;                     % Element spacing in wavelengths
theta = -90:0.1:90;          % Scan angles
theta0 = 10;                 % Desired signal direction
theta_jam = 45;              % Interference direction
snapshots = 1000;            % Number of snapshots
SNR_dB = 20;                 % Desired signal power in dB
INR_dB = 30;                 % Interference power in dB
alpha = 0.1;                 % Diagonal loading factor
mu = 0.01;                   % LMS step size

% Steering vector function
a = @(theta_deg) exp(1j * 2 * pi * d * (0:N-1)' * sind(theta_deg));

% Generate signals
s = sqrt(10^(SNR_dB/10)) * (randn(1,snapshots) + 1j*randn(1,snapshots)) / sqrt(2);  % Desired
j = sqrt(10^(INR_dB/10)) * (randn(1,snapshots) + 1j*randn(1,snapshots)) / sqrt(2);  % Jammer
noise = (randn(N, snapshots) + 1j * randn(N, snapshots)) / sqrt(2);                % Noise

% Array responses
a_s = a(theta0);      % Steering vector to desired
a_j = a(theta_jam);   % Steering vector to jammer

% Total received signal
X = a_s * s + a_j * j + noise;

% Interference + noise only (for SMI)
X_jn = a_j * j + noise;
R_smi = (X_jn * X_jn') / snapshots;
R_dl = R_smi + alpha * eye(N);

% Full covariance for ideal MVDR
R_full = cov(X.');  % True full covariance matrix

% MVDR weight
w_mvdr = (R_full \ a_s) / (a_s' * (R_full \ a_s));

% SMI weight (clean estimate)
w_smi = (R_smi \ a_s) / (a_s' * (R_smi \ a_s));

% DL-SMI weight
w_dl = (R_dl \ a_s) / (a_s' * (R_dl \ a_s));

% LMS weight
w_lms = zeros(N,1);
for n = 1:snapshots
    x_n = X(:,n);
    e_n = s(n) - w_lms' * x_n;
    w_lms = w_lms + mu * x_n * conj(e_n);
end

% Beampattern computation
P_mvdr = zeros(size(theta));
P_smi  = zeros(size(theta));
P_dl   = zeros(size(theta));
P_lms  = zeros(size(theta));

for i = 1:length(theta)
    a_theta = a(theta(i));
    P_mvdr(i) = abs(w_mvdr' * a_theta)^2;
    P_smi(i)  = abs(w_smi'  * a_theta)^2;
    P_dl(i)   = abs(w_dl'   * a_theta)^2;
    P_lms(i)  = abs(w_lms'  * a_theta)^2;
end

% Convert to dB and normalize
epsilon = 1e-12;
P_mvdr = 10*log10(max(P_mvdr, epsilon) / max(P_mvdr));
P_smi  = 10*log10(max(P_smi,  epsilon) / max(P_mvdr));
P_dl   = 10*log10(max(P_dl,   epsilon) / max(P_mvdr));
P_lms  = 10*log10(max(P_lms,  epsilon) / max(P_mvdr));

% Plot
figure;
plot(theta, P_mvdr, 'b-', 'LineWidth', 1.5); hold on;
plot(theta, P_smi, 'r--', 'LineWidth', 1.5);
plot(theta, P_dl, 'k-.', 'LineWidth', 1.5);
plot(theta, P_lms, 'g:', 'LineWidth', 1.5);
xlabel('Angle (degrees)');
ylabel('Normalized Power (dB)');
title('Clean Beampattern Comparison');
legend('MVDR (Ideal)', 'SMI (clean)', 'SMI + DL', 'LMS');
grid on;
axis([-90 90 -60 0]);
