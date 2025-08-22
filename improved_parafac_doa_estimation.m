
%===================== Improved 2D DoA Estimation with PARAFAC and MUSIC =====================%
clear; clc; close all;

%----------------------- UPA Parameters -----------------------%
Mx = 6; My = 6;               % UPA dimensions
dx = 0.5; dy = 0.5;           % element spacing (in wavelengths)
N = Mx * My;
K = 3;                        % number of sources
T = 200;                      % snapshots
SNR = 20;                     % dB

%----------------------- True Source Directions -----------------------%
az_true = [-20, 30, 60];      % azimuth angles in degrees
el_true = [20, 10, 40];       % elevation angles in degrees
az_rad = deg2rad(az_true);
el_rad = deg2rad(el_true);

%----------------------- Steering Matrix Generation -----------------------%
[x_grid, y_grid] = meshgrid(0:Mx-1, 0:My-1);
x_grid = x_grid(:); y_grid = y_grid(:);
A = zeros(N, K);

for k = 1:K
    u = sin(el_rad(k)) * cos(az_rad(k));
    v = sin(el_rad(k)) * sin(az_rad(k));
    A(:,k) = exp(1j*2*pi*(dx*x_grid*u + dy*y_grid*v));
end

%----------------------- Simulate Source Signals -----------------------%
S = randn(K, T) + 1j*randn(K, T);
X = A * S;

%----------------------- Add Noise -----------------------%
sigma = norm(X, 'fro') / sqrt(N*T*10^(SNR/10));
Xn = X + sigma*(randn(N,T) + 1j*randn(N,T))/sqrt(2);

%----------------------- Tensor Formation -----------------------%
X_tensor = reshape(Xn, [Mx, My, T]);

%----------------------- SVD Initialization -----------------------%
X1 = reshape(X_tensor, Mx, []);      % mode-1 unfolding
X2 = reshape(permute(X_tensor, [2 1 3]), My, []); % mode-2
X3 = reshape(permute(X_tensor, [3 1 2]), T, []);  % mode-3

[U1,~,~] = svd(X1, 'econ');
[U2,~,~] = svd(X2, 'econ');
[U3,~,~] = svd(X3, 'econ');

Ax = U1(:,1:K);
Ay = U2(:,1:K);
At = U3(:,1:K);

%----------------------- PARAFAC via ALS -----------------------%
max_iter = 50;
for iter = 1:max_iter
    % Update Ax
    for i = 1:Mx
        V = zeros(K,K); Z = zeros(K,1);
        for t = 1:T
            Yt = squeeze(X_tensor(i,:,t)).';
            Bt = Ay .* At(t,:); 
            V = V + Bt'*Bt;
            Z = Z + Bt'*Yt;
        end
        Ax(i,:) = (V \ Z).';
    end

    % Update Ay
    for j = 1:My
        V = zeros(K,K); Z = zeros(K,1);
        for t = 1:T
            Xt = squeeze(X_tensor(:,j,t));
            Bt = Ax .* At(t,:); 
            V = V + Bt'*Bt;
            Z = Z + Bt'*Xt;
        end
        Ay(j,:) = (V \ Z).';
    end

    % Update At
    for t = 1:T
        Xt = squeeze(X_tensor(:,:,t));
        Bt = Ax.' * Xt * Ay;
        At(t,:) = diag(Bt).';
    end
end

%----------------------- MUSIC Algorithm -----------------------%
Rxx = (Xn * Xn') / T;
[U, D] = eig(Rxx);
[eigenvals, idx] = sort(diag(D), 'descend');
U = U(:,idx);
Un = U(:,K+1:end); 

%----------------------- 2D Angle Grid -----------------------%
angles = -90:1:90;
[AzGrid, ElGrid] = meshgrid(angles, angles);
P_music = zeros(size(AzGrid));
P_parafac = zeros(size(AzGrid));

%----------------------- Evaluate MUSIC and PARAFAC Spectra -----------------------%
for m = 1:numel(AzGrid)
    az0 = deg2rad(AzGrid(m));
    el0 = deg2rad(ElGrid(m));
    u = sin(el0)*cos(az0);
    v = sin(el0)*sin(az0);

    a_x = exp(1j*2*pi*dx*(0:Mx-1)'*u);
    a_y = exp(1j*2*pi*dy*(0:My-1)'*v);
    a_full = exp(1j*2*pi*(dx*x_grid*u + dy*y_grid*v));

    % MUSIC
    P_music(m) = 1 / abs(a_full' * (Un*Un') * a_full);

    % PARAFAC (project Ax and Ay onto steering vectors)
    temp = 0;
    for k = 1:K
        temp = temp + abs((Ax(:,k)'*a_x)*(Ay(:,k)'*a_y));
    end
    P_parafac(m) = temp;
end

%----------------------- Plot Spectra -----------------------%
figure;
subplot(1,2,1);
imagesc(angles, angles, abs(P_music)); axis xy;
title('MUSIC Spectrum'); xlabel('Azimuth'); ylabel('Elevation'); colorbar;
hold on;
plot(az_true, el_true, 'wx', 'LineWidth', 2, 'MarkerSize', 10);

subplot(1,2,2);
imagesc(angles, angles, abs(P_parafac)); axis xy;
title('PARAFAC Spectrum'); xlabel('Azimuth'); ylabel('Elevation'); colorbar;
hold on;
plot(az_true, el_true, 'wx', 'LineWidth', 2, 'MarkerSize', 10);
