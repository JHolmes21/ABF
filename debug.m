
% Diagnostic Beampattern Comparison: MVDR, SMI, DL-SMI, LMS
clear; clc;

%% Parameters
N = 10;
d = 0.5;
theta_scan = -90:0.1:90;
snapshots = 2000;
theta_desired = 10;
theta_jammer = 45;
SNR_dB = 40;
INR_dB = 30;
alpha = 0.1;
mu = 0.05;

a = @(theta) exp(1j*2*pi*d*(0:N-1)'*sind(theta));

%% Generate signals
s = sqrt(10^(SNR_dB/10)) * (randn(1, snapshots) + 1j * randn(1, snapshots)) / sqrt(2);
j = sqrt(10^(INR_dB/10)) * (randn(1, snapshots) + 1j * randn(1, snapshots)) / sqrt(2);
noise = (randn(N, snapshots) + 1j * randn(N, snapshots)) / sqrt(2);

a_s = a(theta_desired);
a_j = a(theta_jammer);

X_full = a_s * s + a_j * j + noise;
X_jn = a_j * j + noise;

%% Covariance matrices
R_mvdr = (X_full * X_full') / snapshots;
R_smi = (X_jn * X_jn') / snapshots;
R_dl = R_smi + alpha * eye(N);

%% Weights
w_mvdr = (R_mvdr \ a_s) / (a_s' * (R_mvdr \ a_s));
w_smi  = (R_smi  \ a_s) / (a_s' * (R_smi  \ a_s));
w_dl   = (R_dl   \ a_s) / (a_s' * (R_dl   \ a_s));

%% LMS
w_lms = zeros(N,1);
for n = 1:snapshots
    x_n = X_full(:,n);
    e = s(n) - w_lms' * x_n;
    w_lms = w_lms + mu * x_n * conj(e);
end

%% Beampatterns (linear)
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

%% Print max power values (linear scale)
fprintf('Max Power (Linear):\n');
fprintf('MVDR: %.4f\n', max(P_mvdr));
fprintf('SMI : %.4f\n', max(P_smi));
fprintf('DL  : %.4f\n', max(P_dl));
fprintf('LMS : %.4f\n', max(P_lms));

%% Plot raw power (linear)
figure;
plot(theta_scan, P_mvdr, 'b-', 'LineWidth', 1.5); hold on;
plot(theta_scan, P_smi,  'r--', 'LineWidth', 1.5);
plot(theta_scan, P_dl,   'k-.', 'LineWidth', 1.5);
plot(theta_scan, P_lms,  'g:',  'LineWidth', 1.5);
xlabel('Angle (degrees)');
ylabel('Array Output Power (Linear)');
title('Raw Beampatterns (Linear Scale)');
legend('MVDR', 'SMI', 'DL-SMI', 'LMS');
grid on;
axis([-90 90 0 inf]);
