%% parafac_vs_music_circular_array_2D_robust.m
% Robust PARAFAC vs MUSIC DOA (2D az–el) for a UCA.
% Upgrades: diagonal masking, noise subtraction, overlapped Hann blocks,
% MUSIC-seeded init, coarse→fine grid, phase anchoring.

clear; clc; close all;

%% ===================== USER SETTINGS =====================
fc       = 3.5e9;    c=3e8;  lambda=c/fc;
M        = 16;       radius=0.5*lambda;

K        = 3;
az_deg   = [-20, 15, 60];          % true az
el_deg   = [ 10,  0,-10];          % true el

Nsnap    = 8000;                   % total snapshots
blockLen = 256;                    % snapshots per block
overlap  = 0.5;                    % 0..0.9
use_hann = true;

SNR_dB   = [10, 5, 0];
source_type='qpsk'; fs=50e3; tone_offsets_Hz=[2e3,4e3,7e3];

% MUSIC coarse/fine grids
AZ_coarse = -90:1.0:90;   EL_coarse = -30:1.0:30;
AZ_fine   = -1.5:0.1:1.5; EL_fine   = -1.0:0.1:1.0;

% PARAFAC options
cp_maxit = 300; cp_tol = 1e-6; cp_verbose = true;
gamma_damp = 0.7;   % ALS damping (0<gamma<=1)
mask_diagonal = true;
subtract_noise_floor = true;
rng(7);
% ==========================================================

%% ======== Geometry & steering ==========================================
k0=2*pi/lambda;
m=(0:M-1).'; phi_m=2*pi*m/M;
r_m=[radius*cos(phi_m), radius*sin(phi_m), zeros(M,1)];

a_true = @(az,el) exp(1j*k0*(r_m*[cosd(el)*cosd(az); cosd(el)*sind(az); sind(el)]));
Atrue=zeros(M,K);
for k=1:K, Atrue(:,k)=a_true(az_deg(k),el_deg(k)); end

%% ======== Simulate data =================================================
switch lower(source_type)
  case 'qpsk'
    S=(sign(randn(K,Nsnap))+1j*sign(randn(K,Nsnap)))/sqrt(2);
  case 'tone'
    t=(0:Nsnap-1)/fs;  S=zeros(K,Nsnap);
    for k=1:K, S(k,:)=exp(1j*2*pi*tone_offsets_Hz(k)*t); end
  otherwise, error('unknown source_type');
end
S = diag( sqrt(10.^(SNR_dB(:).'/10)) ) * S;
N = (randn(M,Nsnap)+1j*randn(M,Nsnap))/sqrt(2);
X = Atrue*S + N;

%% ======== Overlapped, tapered covariance slices ========================
hop = max(1, round(blockLen*(1-overlap)));
starts = 1:hop:(Nsnap-blockLen+1);
B = numel(starts);
R = zeros(M,M,B);
w = ones(1,blockLen); if use_hann, w = hann(blockLen).'; end
wn = w/sqrt(mean(w.^2));  % preserve average power
for b=1:B
    idx = starts(b)+(0:blockLen-1);
    Xb = X(:,idx).*wn;                   % per-sample taper
    R(:,:,b) = (Xb*Xb')/blockLen;
end

%% ======== Estimate & subtract noise floor; mask diagonal =================
Rproc = R;
if subtract_noise_floor
    % Estimate sigma^2 per slice from smallest eigenvalues (median)
    for b=1:B
        ev = sort(real(eig((R(:,:,b)+R(:,:,b)')/2)),'ascend');
        sig2 = median(ev( max(1,M-3):M ));     % robust-ish
        Rproc(:,:,b) = R(:,:,b) - sig2*eye(M);
    end
end
if mask_diagonal
    for b=1:B, Rproc(1:M+1:end,b) = 0; end % zero diagonals in data tensor
end

%% ======== MUSIC (reference) ============================================
Rfull=(X*X')/Nsnap;
[Ev,D]=eig((Rfull+Rfull')/2);
[~,ix]=sort(real(diag(D)),'descend'); Ev=Ev(:,ix); En=Ev(:,K+1:end);

% MUSIC helper (coarse→fine)
[AZm,ELm]=meshgrid(AZ_coarse,EL_coarse); Na=numel(AZ_coarse); Ne=numel(EL_coarse);
dict = @(az,el) normalize_col(a_true(az,el));
Pcoarse = zeros(Ne,Na);
for ii=1:Ne, for jj=1:Na
    a = dict(AZm(ii,jj),ELm(ii,jj));
    Pcoarse(ii,jj) = 1/max(norm(En'*a)^2,1e-12);
end, end
est_music = pick_topk_2d(Pcoarse, AZ_coarse, EL_coarse, K);

% refine each peak locally
for k=1:numel(est_music.az)
    azc=est_music.az(k); elc=est_music.el(k);
    az_loc = azc + AZ_fine; el_loc = elc + EL_fine;
    [AZl,ELl]=meshgrid(az_loc, el_loc);
    Pf = zeros(numel(el_loc), numel(az_loc));
    for ii=1:numel(el_loc), for jj=1:numel(az_loc)
        Pf(ii,jj) = 1/max(norm(En'*dict(AZl(ii,jj),ELl(ii,jj)))^2,1e-12);
    end, end
    pk = pick_topk_2d(Pf, az_loc, el_loc, 1);
    est_music.az(k)=pk.az(1); est_music.el(k)=pk.el(1);
end

%% ======== PARAFAC (CP-ALS) with coupling & masking ======================
% Build tensor
Xten = Rproc;  % M x M x B
% MUSIC-seeded initialization: match K largest peaks to dictionary columns
% Build a modest dictionary to initialize
AZi = -90:2:90; ELi = -30:2:30;
Agrid = zeros(M, numel(AZi)*numel(ELi));
idx=1;
for ee=1:numel(ELi)
  for aa=1:numel(AZi)
    Agrid(:,idx) = dict(AZi(aa), ELi(ee)); idx=idx+1;
  end
end
% Use Ravg eigenvectors -> correlate to dictionary -> pick K seeds
Ravg = mean(Rproc,3);
[U0,~,~]=svd((Ravg+Ravg')/2,'econ');
scores = abs(Agrid'*(U0(:,1:K)));
[~,imax] = maxk(sum(scores,2), K);
A = Agrid(:,imax);              % seed mode-1 with grid steering
A = phase_anchor(A);            % fix column phase (stable)
Bf = conj(A);
C  = randn(B,K)+1j*randn(B,K);

fit_prev=0;
for it=1:cp_maxit
    % --- Update A ---
    Z = khatri_rao(C, Bf);                 % (B*M) x K
    X1 = reshape(Xten, M, []);             % M x (M*B)
    A_ls = X1 / (Z.');
    A = (1-gamma_damp)*A + gamma_damp*A_ls;
    A = phase_anchor(A);
    % --- Update B; then tie ---
    Z = khatri_rao(C, A);
    X2 = reshape(permute(Xten,[2 1 3]), M, []);  % M x (M*B)
    B_ls = X2 / (Z.');
    % align phase to A, then hard tie
    for k=1:K
        ph = angle(sum(conj(B_ls(:,k)).*A(:,k)));
        B_ls(:,k) = exp(1j*ph)*B_ls(:,k);
    end
    Bf = conj(A);
    % --- Update C ---
    Z = khatri_rao(Bf, A);                 % (M*M) x K
    X3 = reshape(permute(Xten,[3 1 2]), B, []); % B x (M*M)
    C_ls = X3 / (Z.');
    C = (1-gamma_damp)*C + gamma_damp*C_ls;
    % --- Normalize; compute masked fit ---
    [A,Bf,C] = normalize_columns_abc(A,Bf,C);
    num=0; den=0;
    for b=1:B
        Xhat = A*diag(C(b,:))*Bf';
        if mask_diagonal
            Xb = Xten(:,:,b); Xb(1:M+1:end)=0;
            Xhat(1:M+1:end)=0;
        else
            Xb = Xten(:,:,b);
        end
        num = num + norm(Xb - Xhat,'fro')^2;
        den = den + norm(Xb,'fro')^2;
    end
    fit = 1 - num/max(den,eps);
    if cp_verbose && (mod(it,10)==0 || it==1)
        fprintf('CP-ALS iter %3d, fit = %.6f\n', it, fit);
    end
    if abs(fit-fit_prev) < cp_tol, break; end
    fit_prev = fit;
end

%% ======== 2D grid matching (coarse→fine) of PARAFAC columns ============
est_par = struct('az',[],'el',[]);
for k=1:K
    ak = normalize_col(A(:,k));
    % coarse
    best = [-Inf, NaN, NaN];
    for ee=1:numel(EL_coarse), for aa=1:numel(AZ_coarse)
        val = abs( ak' * dict(AZ_coarse(aa), EL_coarse(ee)) );
        if val>best(1), best=[val, AZ_coarse(aa), EL_coarse(ee)]; end
    end, end
    % fine around best
    az_loc = best(2) + AZ_fine; el_loc = best(3) + EL_fine;
    best2 = best;
    for ee=1:numel(el_loc), for aa=1:numel(az_loc)
        val = abs( ak' * dict(az_loc(aa), el_loc(ee)) );
        if val>best2(1), best2=[val, az_loc(aa), el_loc(ee)]; end
    end, end
    est_par.az(k)=best2(2); est_par.el(k)=best2(3);
end

%% ================= Results & visualization =============================
true_sorted    = sortrows([az_deg(:), el_deg(:)],[1 2]);
music_sorted   = sortrows([est_music.az(:), est_music.el(:)],[1 2]);
parafac_sorted = sortrows([[est_par.az].', [est_par.el].'],[1 2]);

fprintf('\n=== RESULTS (Az, El in deg) ===\n');
fprintf('True:      %s\n', mat2str(true_sorted));
fprintf('MUSIC:     %s\n', mat2str(music_sorted));
fprintf('PARAFAC:   %s\n', mat2str(parafac_sorted));

% MUSIC coarse map (for reference)
[AZm,ELm]=meshgrid(AZ_coarse,EL_coarse);
Pcoarse_dB = 10*log10(Pcoarse / max(Pcoarse,[],'all'));
figure('Name','MUSIC 2D Coarse Pseudospectrum');
imagesc(AZ_coarse, EL_coarse, Pcoarse_dB); axis xy; colorbar;
xlabel('Az [deg]'); ylabel('El [deg]'); title('MUSIC (coarse) [dB]');
hold on;
plot(az_deg,el_deg,'wx','MarkerSize',10,'LineWidth',2);
plot(music_sorted(:,1),music_sorted(:,2),'wo','MarkerSize',8,'LineWidth',1.5);
plot(parafac_sorted(:,1), parafac_sorted(:,2), 'w+','MarkerSize',10,'LineWidth',1.5);
legend('True','MUSIC est','PARAFAC est','Location','southoutside');

%% ================= Helper functions ====================================
function v = normalize_col(v), v = v / max(norm(v),eps); end
function A = phase_anchor(A)
% Make first sensor real-positive for each column (anchors arbitrary phase)
  for k=1:size(A,2)
    ph = angle(A(1,k));
    A(:,k) = A(:,k) * exp(-1j*ph);
  end
end
function Z = khatri_rao(A,B)
  [IA,K]=size(A); [IB,KB]=size(B); if KB~=K, error('KR col mismatch'); end
  Z=zeros(IA*IB,K);
  for k=1:K, Z(:,k)=kron(A(:,k),B(:,k)); end
end
function [A,B,C]=normalize_columns_abc(A,B,C)
  K=size(A,2);
  for k=1:K
    na=norm(A(:,k)); nb=norm(B(:,k));
    if na==0, na=1; end; if nb==0, nb=1; end
    s=sqrt(na*nb);
    A(:,k)=A(:,k)/s; B(:,k)=B(:,k)/s; C(:,k)=C(:,k)*(s^2);
  end
end
function est = pick_topk_2d(map, az_grid_deg, el_grid_deg, K)
  map2=map; est.az=zeros(1,K); est.el=zeros(1,K);
  nbh_az=max(1,round(2/mean(diff(az_grid_deg))));
  nbh_el=max(1,round(2/mean(diff(el_grid_deg))));
  for k=1:K
    [val,idx]=max(map2(:)); if ~isfinite(val)||val<=0, est.az=est.az(1:k-1); est.el=est.el(1:k-1); break; end
    [ii,jj]=ind2sub(size(map2),idx);
    est.az(k)=az_grid_deg(jj); est.el(k)=el_grid_deg(ii);
    i1=max(1,ii-nbh_el); i2=min(size(map2,1),ii+nbh_el);
    j1=max(1,jj-nbh_az); j2=min(size(map2,2),jj+nbh_az);
    map2(i1:i2,j1:j2)=-Inf;
  end
end
