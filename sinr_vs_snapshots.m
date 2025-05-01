% SINR vs. Snapshot Number for MVDR, SMI, DL-SMI
clear; clc;

N = 10; d = 0.5;
theta0 = 10; theta_jam = 45;
a = @(theta) exp(1j*2*pi*d*(0:N-1)'*sind(theta));
snap_range = 10:10:500;
SNR_dB = 20; INR_dB = 30;

SINR_mvdr = zeros(size(snap_range));
SINR_smi = zeros(size(snap_range));
SINR_dl = zeros(size(snap_range));

for k = 1:length(snap_range)
    snapshots = snap_range(k);
    s = sqrt(10^(SNR_dB/10)) * (randn(1,snapshots)+1j*randn(1,snapshots))/sqrt(2);
    j = sqrt(10^(INR_dB/10)) * (randn(1,snapshots)+1j*randn(1,snapshots))/sqrt(2);
    noise = (randn(N,snapshots)+1j*randn(N,snapshots))/sqrt(2);
    X = a(theta0)*s + a(theta_jam)*j + noise;

    R_hat = (X*X')/snapshots;
    R_loaded = R_hat + 0.1*eye(N);
    R_true = cov(X.');

    w_mvdr = (R_true\a(theta0)) / (a(theta0)' * (R_true\a(theta0)));
    w_smi = (R_hat\a(theta0)) / (a(theta0)' * (R_hat\a(theta0)));
    w_dl  = (R_loaded\a(theta0)) / (a(theta0)' * (R_loaded\a(theta0)));

    desired = a(theta0)*s;
    jammer_noise = a(theta_jam)*j + noise;

    SINR_mvdr(k) = var(w_mvdr'*desired) / var(w_mvdr'*jammer_noise);
    SINR_smi(k)  = var(w_smi'*desired)  / var(w_smi'*jammer_noise);
    SINR_dl(k)   = var(w_dl'*desired)   / var(w_dl'*jammer_noise);
end

% Plot
figure;
plot(snap_range, 10*log10(SINR_mvdr), 'b-', ...
     snap_range, 10*log10(SINR_smi), 'r--', ...
     snap_range, 10*log10(SINR_dl), 'k-.', 'LineWidth', 1.5);
xlabel('Number of Snapshots'); ylabel('Output SINR (dB)');
title('SINR vs. Number of Snapshots');
legend('MVDR (Ideal)', 'SMI', 'SMI + DL');
grid on;
