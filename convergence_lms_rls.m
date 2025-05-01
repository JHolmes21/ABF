% LMS vs RLS Convergence Over Time
clear; clc;

N = 10; d = 0.5;
theta0 = 10; theta_jam = 45;
snapshots = 300;
a = @(theta) exp(1j*2*pi*d*(0:N-1)'*sind(theta));
SNR_dB = 20; INR_dB = 30;

s = sqrt(10^(SNR_dB/10)) * (randn(1,snapshots)+1j*randn(1,snapshots))/sqrt(2);
j = sqrt(10^(INR_dB/10)) * (randn(1,snapshots)+1j*randn(1,snapshots))/sqrt(2);
noise = (randn(N,snapshots)+1j*randn(N,snapshots))/sqrt(2);
X = a(theta0)*s + a(theta_jam)*j + noise;

desired = a(theta0)*s;
interference = a(theta_jam)*j + noise;

% LMS
mu = 0.01;
w_lms = zeros(N,1);
SINR_lms = zeros(1,snapshots);

% RLS
delta = 0.01; lambda = 0.99;
P = eye(N)/delta;
w_rls = zeros(N,1);
SINR_rls = zeros(1,snapshots);

for n = 1:snapshots
    x_n = X(:,n); d_n = s(n);

    % LMS
    e_lms = d_n - w_lms' * x_n;
    w_lms = w_lms + mu * x_n * conj(e_lms);
    SINR_lms(n) = var(w_lms'*desired(:,1:n)) / var(w_lms'*interference(:,1:n));

    % RLS
    k = (P * x_n) / (lambda + x_n' * P * x_n);
    e_rls = d_n - w_rls' * x_n;
    w_rls = w_rls + k * conj(e_rls);
    P = (P - k * x_n' * P) / lambda;
    SINR_rls(n) = var(w_rls'*desired(:,1:n)) / var(w_rls'*interference(:,1:n));
end

% Plot
figure;
plot(1:snapshots, 10*log10(SINR_lms), 'g--', ...
     1:snapshots, 10*log10(SINR_rls), 'm-', 'LineWidth', 1.5);
xlabel('Iteration'); ylabel('Output SINR (dB)');
title('LMS vs RLS Convergence');
legend('LMS', 'RLS'); grid on;
