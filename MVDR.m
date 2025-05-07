% Parameters
M = 10;                    % Number of sensors
d = 0.5;                   % Element spacing in wavelengths
theta_desired = 30;        % Desired signal direction in degrees
theta_desired_rad = deg2rad(theta_desired);
N_snapshots = 200;         % Number of time samples

% Steering vector function
steervec = @(theta_rad) exp(1j*2*pi*d*(0:M-1)'*sin(theta_rad));

% Generate synthetic data
SNR = 20;                  % Desired signal SNR (dB)
INR = 30;                  % Interference-to-noise ratio (dB)
theta_interf = 60;         % Interferer direction in degrees

a_desired = steervec(theta_desired_rad);
a_interf = steervec(deg2rad(theta_interf));

s = sqrt(10^(SNR/10)) * randn(1, N_snapshots);          % Desired signal
i = sqrt(10^(INR/10)) * randn(1, N_snapshots);          % Interference
n = randn(M, N_snapshots);                              % Noise

X = a_desired * s + a_interf * i + n;                   % Received signal matrix

% Estimate covariance matrix
R = (X * X') / N_snapshots;

% MVDR weights
w_mvdr = (R \ a_desired) / (a_desired' * (R \ a_desired));

% Output signal
y = w_mvdr' * X;

% Plot beampattern
theta_scan = -90:0.5:90;
beampattern = zeros(size(theta_scan));
for k = 1:length(theta_scan)
    a_scan = steervec(deg2rad(theta_scan(k)));
    beampattern(k) = abs(w_mvdr' * a_scan);
end
beampattern = beampattern / max(beampattern); % Normalize

figure;
plot(theta_scan, 20*log10(beampattern));
xlabel('Angle (degrees)');
ylabel('Beampattern (dB)');
title('MVDR Beamformer Beampattern');
grid on;
ylim([-60 0]);
