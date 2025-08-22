
%==================== Accurate DoA Estimation Using PARAFAC and MUSIC ====================%
% Uses: base MATLAB only, 2D UPA, uncorrelated sources
clear; clc; close all;

%-------------------- UPA and Signal Parameters --------------------%
Mx = 6; My = 6;
dx = 0.5; dy = 0.5;
N = Mx * My;
K = 3;
T = 200;
SNR = 20;

% True emitter directions
az_true = [-20, 30, 60]; % degrees
el_true = [20, 10, 40];  % degrees
az_rad = deg2rad(az_true);
el_rad = deg2rad(el_true);

% Grid index for UPA
[xg, yg] = meshgrid(0:Mx-1, 0:My-1);
xg = xg(:); yg = yg(:);

%-------------------- Steering Matrix A --------------------%
A = zeros(N, K);
for k = 1:K
    u = sin(el_rad(k)) * cos(az_rad(k));
    v = sin(el_rad(k)) * sin(az_rad(k));
    A(:,k) = exp(1j*2*pi*(dx*xg*u + dy*yg*v));
end

% Source signals and noise
S = randn(K, T) + 1j*randn(K, T);
X = A * S;
sigma = norm(X, 'fro') / sqrt(N*T*10^(SNR/10));
Xn = X + sigma*(randn(N,T) + 1j*randn(N,T))/sqrt(2);

%-------------------- Tensor Construction [Mx, My, T] --------------------%
X_tensor = reshape(Xn, [Mx, My, T]);

%-------------------- PARAFAC Decomposition via ALS --------------------%
Ax = randn(Mx, K) + 1j*randn(Mx, K);
Ay = randn(My, K) + 1j*randn(My, K);
At = randn(T, K) + 1j*randn(T, K);
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

%-------------------- Estimate DoAs from PARAFAC Factors --------------------%
angles = -90:1:90;
[AZ, EL] = meshgrid(angles, angles);
num_grid = numel(AZ);
DoAs_est_parafac = zeros(K, 2);

for k = 1:K
    best_score = -inf;
    for idx = 1:num_grid
        az0 = deg2rad(AZ(idx));
        el0 = deg2rad(EL(idx));
        u = sin(el0)*cos(az0);
        v = sin(el0)*sin(az0);
        a_x = exp(1j*2*pi*dx*(0:Mx-1)'*u);
        a_y = exp(1j*2*pi*dy*(0:My-1)'*v);
        score = abs((a_x'*Ax(:,k)) * (a_y'*Ay(:,k)));
        if score > best_score
            best_score = score;
            DoAs_est_parafac(k,:) = [rad2deg(az0), rad2deg(el0)];
        end
    end
end

%-------------------- MUSIC DoA Estimation --------------------%
Rxx = (Xn * Xn') / T;
[U,D] = eig(Rxx);
[eigs_sorted, idx] = sort(diag(D), 'descend');
Un = U(:, idx(K+1:end));

P_music = zeros(size(AZ));
for i = 1:num_grid
    az0 = deg2rad(AZ(i));
    el0 = deg2rad(EL(i));
    u = sin(el0)*cos(az0);
    v = sin(el0)*sin(az0);
    a = exp(1j*2*pi*(dx*xg*u + dy*yg*v));
    P_music(i) = 1 / abs(a' * (Un*Un') * a);
end

%-------------------- Plot MUSIC Spectrum --------------------%
figure;
imagesc(angles, angles, abs(P_music)); axis xy;
xlabel('Azimuth (°)'); ylabel('Elevation (°)');
title('MUSIC Spectrum'); colorbar;
hold on;
plot(az_true, el_true, 'wx', 'LineWidth', 2, 'MarkerSize', 10);

%-------------------- Print Results --------------------%
disp('True DoAs:');
disp([az_true(:), el_true(:)]);

disp('PARAFAC Estimated DoAs:');
disp(DoAs_est_parafac);
