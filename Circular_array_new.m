%% simulate_circular_array_radar.m
% Circular phased array radar receiving multiple narrowband sources
% - Uniform circular array (UCA) in the x-y plane
% - Far-field, narrowband, baseband model
% - Complex baseband sources (option: sinusoids or random QPSK-like symbols)
% - Additive white Gaussian noise
% - MUSIC azimuth-only DOA estimation
% Base MATLAB only.

clear; clc; close all;

%% ===================== USER SETTINGS =====================
fc       = 3.5e9;        % Carrier frequency [Hz]
c        = 3e8;          % Speed of light [m/s]
lambda   = c/fc;         % Wavelength [m]

M        = 16;           % Number of array elements (uniform circular array)
radius   = 0.5*lambda;   % Array radius [m] (≈ lambda/2 is a decent starting point)

K        = 3;                            % Number of sources
az_deg   = [-20, 15, 60];                % Source azimuths [deg], 0 = +x axis, CCW toward +y
el_deg   = [0, 0, 0];                    % Source elevations [deg]; 0 = in-plane
assert(numel(az_deg)==K && numel(el_deg)==K,'az_deg/el_deg must have length K');

Nsnap    = 2000;        % Snapshots (time samples)
SNR_dB   = [10, 5, 0];  % SNR per source (power ratio per sensor, dB). Same length as K.

source_type = 'qpsk';   % 'qpsk' (random symbols) or 'tone'
tone_offsets_Hz = [2e3, 4e3, 7e3];  % Only used if source_type='tone'
fs       = 50e3;        % "Sampling rate" for baseband sim (tone spacing, not strictly needed)

rng(7);                 % Reproducibility
plot_geometry  = true;
plot_music     = true;
scan_grid_deg  = -90:0.25:90;   % MUSIC scan grid (azimuth) for visualization
% ===========================================================

%% Derived values
k0 = 2*pi/lambda;                     % Wavenumber magnitude
az = deg2rad(az_deg(:).');            % 1xK
el = deg2rad(el_deg(:).');            % 1xK

% Array element positions (UCA in x-y plane, centered at origin)
m = (0:M-1).';                        % Mx1
phi_m = 2*pi*m/M;                     % element angular positions
r_m = [radius*cos(phi_m), radius*sin(phi_m), zeros(M,1)];   % Mx3 positions

% Steering matrix A (MxK)
A = zeros(M, K);
for k = 1:K
    % Source direction unit vector (az, el): az from +x toward +y; el from array plane upward
    u = [cos(el(k))*cos(az(k));  % x
         cos(el(k))*sin(az(k));  % y
         sin(el(k))];            % z
    % Phase at each sensor: exp(j*k0 * r_m · u)
    A(:,k) = exp(1j * k0 * (r_m * u));
end

%% Generate source signals S (K x Nsnap)
switch lower(source_type)
    case 'qpsk'
        % Random QPSK-like symbols with unit power
        S = (sign(randn(K,Nsnap)) + 1j*sign(randn(K,Nsnap))) / sqrt(2);
    case 'tone'
        t = (0:Nsnap-1)/fs;
        if numel(tone_offsets_Hz) ~= K
            error('tone_offsets_Hz must have length K for source_type="tone"');
        end
        S = zeros(K, Nsnap);
        for k = 1:K
            S(k,:) = exp(1j*2*pi*tone_offsets_Hz(k)*t);  % unit power complex tone
        end
    otherwise
        error('Unknown source_type: %s', source_type);
end

% Scale each source to desired SNR at *single sensor* before steering
% Noise variance per sensor = 1 by construction below, so set source power accordingly.
SNR_lin = 10.^(SNR_dB(:).'/10);   % 1xK
gain   = sqrt(SNR_lin);           % amplitude scale
S = diag(gain) * S;               % KxNsnap

%% Form array data X = A*S + N  (M x Nsnap)
% White circular complex Gaussian noise with unit variance per sensor
N = (randn(M,Nsnap) + 1j*randn(M,Nsnap))/sqrt(2);
X = A*S + N;

%% Sample covariance matrix (MxM)
Rhat = (X*X')/Nsnap;

%% Estimate number of sources (optional): here we assume known K.
% If you don’t know K, you can inspect the eigenvalue “elbow” or use
% information criteria (AIC/MDL). For simplicity, we use the known K.

% Eigen-decomposition
[Ev, D] = eig((Rhat+Rhat')/2);        % Hermitian safeguard
[eigs_sorted, idx] = sort(real(diag(D)),'descend');
Ev = Ev(:, idx);

Es = Ev(:, 1:K);                       % signal subspace
En = Ev(:, K+1:end);                   % noise subspace

%% MUSIC spectrum over azimuth (elevation fixed at 0 unless changed)
Pmu = zeros(size(scan_grid_deg));
for i = 1:numel(scan_grid_deg)
    az_i = deg2rad(scan_grid_deg(i));
    el_i = 0; % or set another fixed elevation if desired
    a_i = steering_vec_circular(k0, r_m, az_i, el_i);  % Mx1
    denom = norm(En' * a_i)^2;          % ||E_n^H a(θ)||^2
    Pmu(i) = 1 / max(denom, 1e-12);     % guard against divide-by-zero
end
Pmu_dB = 10*log10(Pmu / max(Pmu));

%% ====== Plots ======
if plot_geometry
    figure('Name','Array Geometry'); 
    plot(r_m(:,1), r_m(:,2), 'o', 'MarkerFaceColor',[.2 .6 1], 'MarkerSize',6); grid on; axis equal;
    xlabel('x [m]'); ylabel('y [m]');
    title(sprintf('Uniform Circular Array (M=%d, radius=%.3f m, \\lambda=%.3f m)', M, radius, lambda));
    hold on; plot(0,0,'kx'); legend('Elements','Array center');
end

if plot_music
    figure('Name','MUSIC Azimuth Spectrum');
    plot(scan_grid_deg, Pmu_dB, 'LineWidth',1.5); grid on;
    xlabel('Azimuth [deg]'); ylabel('Pseudo-spectrum [dB]');
    title(sprintf('MUSIC DOA (K=%d, snapshots=%d, SNR_dB=%s)', K, Nsnap, mat2str(SNR_dB)));
    xlim([min(scan_grid_deg), max(scan_grid_deg)]);
    % Mark true DOAs
    hold on;
    yl = ylim;
    for k = 1:K
        xline(az_deg(k),'--','Color',[0.2 0.2 0.2],'LineWidth',1);
        text(az_deg(k), yl(2)-0.05*range(yl), sprintf('%d^\\circ', az_deg(k)), ...
            'HorizontalAlignment','center','VerticalAlignment','top');
    end
end

%% ====== Peak-pick estimated DOAs (simple) ======
% Find local maxima in the pseudo-spectrum; pick K highest peaks with min separation.
[pk, locs] = findpeaks(Pmu_dB, 'SortStr','descend');
est_az = scan_grid_deg(sort(locs(1:min(K,end))));
est_az = sort(est_az(:).');  %#ok<NASGU>

fprintf('True azimuths [deg]:   %s\n', mat2str(sort(az_deg)));
if ~isempty(est_az)
    fprintf('Estimated azimuths [deg] (MUSIC): %s\n', mat2str(est_az));
else
    fprintf('Estimated azimuths: <no peaks found>\n');
end

%% ================== Helper function ==================
function a = steering_vec_circular(k0, r_m, az, el)
% a = steering_vec_circular(k0, r_m, az, el)
% k0 : wavenumber magnitude 2*pi/lambda
% r_m: Mx3 element position vectors [x y z]
% az : azimuth [rad], 0 along +x, increases CCW toward +y
% el : elevation [rad], 0 in array plane, + upward
% Returns Mx1 steering vector a(az,el)
    u = [cos(el)*cos(az); cos(el)*sin(az); sin(el)];  % direction unit vector
    a = exp(1j * k0 * (r_m * u));
end
