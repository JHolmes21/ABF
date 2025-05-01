% Beampattern Comparison: MVDR, SMI, DL-SMI, LMS
clear; clc;

% Array parameters
N = 10;
d = 0.5;
theta = -90:0.1:90;
theta0 = 10;
theta_jam = 45;
SNR_dB = 20;
INR_dB = 30;
snapshots = 200;

a = @(angle_deg) exp(1j*2*pi*d*(0:N-1)'*sind(angle_deg));

s = sqrt(10^(SNR_dB/10)) * (randn(1,snapshots) + 1j*randn(1,snapshots)) / sqrt(2);
j = sqrt(10^(INR_dB/10)) * (randn(1,snapshots) + 1j*randn(1,snapshots)) / sqrt(2);
noise = (randn(N,snapshots) + 1j*randn(N,snapshots)) / sqrt(2);

X = a(theta0) * s + a(theta_jam) * j + noise;

R_hat = (X * X') / snapshots;
R_loaded = R_hat + 0.1 * eye(N);
R_true = cov(X.');

w_mvdr = (R_true\a(theta0)) / (a(theta0)' * (R_true\a(theta0)));
w_smi = (R_hat\a(theta0)) / (a(theta0)' * (R_hat\a(theta0)));
w_dl = (R_loaded\a(theta0)) / (a(theta0)' * (R_loaded\a(theta0)));

% LMS
mu = 0.01;
w_lms = zeros(N,1);
d_desired = s;
for n = 1:snapshots
    x_n = X(:,n);
    e = d_desired(n) - w_lms' * x_n;
    w_lms = w_lms + mu * x_n * conj(e);
end

% Beampatterns
P_mvdr = zeros(size(theta));
P_smi = zeros(size(theta));
P_dl = zeros(size(theta));
P_lms = zeros(size(theta));

for idx = 1:length(theta)
    a_theta = a(theta(idx));
    P_mvdr(idx) = abs(w_mvdr' * a_theta).^2;
    P_smi(idx) = abs(w_smi' * a_theta).^2;
    P_dl(idx) = abs(w_dl' * a_theta).^2;
    P_lms(idx) = abs(w_lms' * a_theta).^2;
end

P_mvdr = 10*log10(P_mvdr / max(P_mvdr));
P_smi = 10*log10(P_smi / max(P_smi));
P_dl = 10*log10(P_dl / max(P_dl));
P_lms = 10*log10(P_lms / max(P_lms));

% Plot
figure;
plot(theta, P_mvdr, 'b-', theta, P_smi, 'r--', theta, P_dl, 'k-.', theta, P_lms, 'g:', 'LineWidth', 1.5);
legend('MVDR (Ideal)', 'SMI', 'SMI + DL', 'LMS');
xlabel('Angle (degrees)'); ylabel('Normalized Power (dB)');
title('Beampattern Comparison');
grid on; axis([-90 90 -60 0]);
