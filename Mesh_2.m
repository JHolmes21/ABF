%% === Demix to get separated signals (like before) ===
AhA = A' * A;
mu  = 1e-3 * trace(AhA)/K;              % ridge for stability
W   = (AhA + mu*eye(K)) \ (A');         % K×m demixer
S_hat = W * X_all;                      % K×Nsnap (estimated components)

%% === Match separated components to true sources (perm + complex scale) ===
% Build normalized correlation matrix Gamma(q,k) = |<S_true_q, S_hat_k>| / (||S_true_q|| ||S_hat_k||)
S_true = Ssig;                          % K×Nsnap (your simulated sources)
num = S_true * S_hat';                  % K×K complex inner products
den = (vecnorm(S_true,2,2)) * (vecnorm(S_hat,2,2)).';   % K×K outer product of norms
Gamma = abs(num) ./ max(den, eps);      % K×K similarity in [0,1]

% Greedy assignment (toolbox-free): find largest remaining correlation
unmatched_true = true(1,K); unmatched_hat = true(1,K);
assign_true = zeros(1,K); assign_hat = zeros(1,K);  % record matches
Gam = Gamma;                                          % working copy
for r = 1:K
    Gam(~unmatched_true, :) = -Inf;                   % mask used rows
    Gam(:, ~unmatched_hat) = -Inf;                    % mask used cols
    [val, idx] = max(Gam(:)); %#ok<ASGLU>
    if ~isfinite(val), break; end
    [iq, ik] = ind2sub(size(Gam), idx);
    assign_true(r) = iq; assign_hat(r) = ik;
    unmatched_true(iq) = false; unmatched_hat(ik) = false;
end
% Trim in case K_est < K or vice versa
valid = assign_true ~= 0 & assign_hat ~= 0;
assign_true = assign_true(valid); assign_hat = assign_hat(valid);
Kmatch = numel(assign_true);

% Compute complex scale for each pair: alpha = <S_true, S_hat> / ||S_hat||^2
S_hat_aligned = zeros(size(S_hat));
metrics = struct('corr',[],'NMSE',[],'SNRdB',[],'alpha',[]);
for t = 1:Kmatch
    q = assign_true(t); k = assign_hat(t);
    alpha = (S_true(q,:) * S_hat(k,:)') / (norm(S_hat(k,:))^2 + eps);
    S_hat_aligned(q,:) = alpha * S_hat(k,:);
    err = S_true(q,:) - S_hat_aligned(q,:);
    corr_qk = abs( (S_true(q,:) * S_hat_aligned(q,:)') / (norm(S_true(q,:))*norm(S_hat_aligned(q,:)) + eps) );
    nmse_qk = norm(err)^2 / (norm(S_true(q,:))^2 + eps);
    snr_qk  = 10*log10( 1 / max(nmse_qk, eps) );   % since NMSE = ||e||^2 / ||true||^2
    metrics.corr(q) = corr_qk;
    metrics.NMSE(q) = nmse_qk;
    metrics.SNRdB(q)= snr_qk;
    metrics.alpha(q)= alpha;
end

%% === Print a tiny report ===
fprintf('\n=== Separation quality (matched pairs) ===\n');
for q = 1:Kmatch
    k = assign_hat(q);
    fprintf('True %d  <--  Est %d : corr=%.3f, NMSE=%.3e, SNR=%.2f dB\n', ...
        assign_true(q), k, metrics.corr(assign_true(q)), metrics.NMSE(assign_true(q)), metrics.SNRdB(assign_true(q)));
end

%% === Plot ALL separated signals on ONE axis (choose real part or magnitude) ===
t = 1:size(S_hat,2);
colors = lines(K);

% ----- Option A: real part overlay (aligned to true) -----
figure('Name','Separated vs True (real part)','Color','w'); hold on;
for q = 1:K
    if any(S_hat_aligned(q,:))
        plot(t, real(S_hat_aligned(q,:)), '-', 'Color', colors(q,:), 'LineWidth', 1.2);
    end
    plot(t, real(S_true(q,:)), '--', 'Color', colors(q,:), 'LineWidth', 1.0);
end
xlabel('Snapshot index n'); ylabel('Real part'); grid on; box on;
title('Separated components (solid) vs True sources (dashed)');
legendStrings = [arrayfun(@(q) sprintf('Est %d', q), 1:K, 'UniformOutput', false), ...
                 arrayfun(@(q) sprintf('True %d', q), 1:K, 'UniformOutput', false)];
legend(legendStrings, 'Location','bestoutside');

% ----- Option B: magnitude overlay (often cleaner for QPSK/tones) -----
figure('Name','Separated vs True (magnitude)','Color','w'); hold on;
for q = 1:K
    if any(S_hat_aligned(q,:))
        plot(t, abs(S_hat_aligned(q,:)), '-', 'Color', colors(q,:), 'LineWidth', 1.2);
    end
    plot(t, abs(S_true(q,:)), '--', 'Color', colors(q,:), 'LineWidth', 1.0);
end
xlabel('Snapshot index n'); ylabel('|·|'); grid on; box on;
title('Separated components (solid) vs True sources (dashed) — magnitude');
legend(legendStrings, 'Location','bestoutside');

%% === Optional: per-component subplots with residuals ===
%{
figure('Name','Per-component overlays with residuals','Color','w');
for q = 1:K
    subplot(K,1,q);
    plot(t, real(S_true(q,:)), '--', 'Color', [0.3 0.3 0.3]); hold on;
    plot(t, real(S_hat_aligned(q,:)), '-', 'Color', colors(q,:), 'LineWidth', 1.2);
    plot(t, real(S_true(q,:) - S_hat_aligned(q,:)), ':', 'Color', [0.2 0.6 1], 'LineWidth', 1.0);
    ylabel(sprintf('q=%d',q)); grid on;
    if q==1, title('Real(true), Real(est aligned), Real(residual)'); end
end
xlabel('Snapshot index n');
%}
