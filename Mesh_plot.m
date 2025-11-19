% === Correlation meshes: how each PARAFAC column "lights up" over (Az,El) ===
azg = AZ_coarse;  elg = EL_coarse;
[AZm, ELm] = meshgrid(azg, elg);   % EL rows x AZ cols
Na = numel(azg); Ne = numel(elg);

for k = 1:K
    ak = A(:,k) / max(norm(A(:,k)), eps);
    corrMap = zeros(Ne, Na);
    for ii = 1:Ne
        for jj = 1:Na
            aij = dict(AZm(ii,jj), ELm(ii,jj));
            corrMap(ii,jj) = abs(aij' * ak);
        end
    end
    figure('Name', sprintf('PARAFAC comp %d: Corr mesh'), 'Color','w');
    mesh(azg, elg, corrMap);  % mesh plot
    xlabel('Azimuth [deg]'); ylabel('Elevation [deg]'); zlabel('|corr|');
    title(sprintf('Component %d: |a^H(az,el) · A(:,%d)|', k, k));
    grid on; view(45,30);
end
