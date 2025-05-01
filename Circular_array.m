% Parameters
N = 10;                          % Number of elements
lambda = 1;                      % Wavelength
k = 2 * pi / lambda;             % Wavenumber
radius = lambda / (2 * sin(pi/N));  % Radius to keep ~λ/2 spacing

% Element positions (circular array in xy-plane)
n = 0:N-1;
phi_n = 2 * pi * n / N;
x_n = radius * cos(phi_n);
y_n = radius * sin(phi_n);
z_n = zeros(1, N);  % flat array

% Beam steering direction (azimuth, elevation in degrees)
steer_az = 30;
steer_el = 20;
steer_theta = deg2rad(steer_az);
steer_phi = deg2rad(steer_el);
ux_steer = sin(steer_phi) * cos(steer_theta);
uy_steer = sin(steer_phi) * sin(steer_theta);

% Spherical scan grid
theta = linspace(-pi, pi, 180);        % Azimuth
phi = linspace(0, pi/2, 90);           % Elevation
[TH, PH] = meshgrid(theta, phi);       % PH = elevation, TH = azimuth

% Direction cosines
ux = sin(PH) .* cos(TH);
uy = sin(PH) .* sin(TH);

% Array Factor calculation
AF = zeros(size(TH));
for i = 1:N
    psi = k * (x_n(i)*(ux - ux_steer) + y_n(i)*(uy - uy_steer));
    AF = AF + exp(1j * psi);
end

AF_mag = abs(AF);
AF_mag = AF_mag / max(AF_mag);  % Normalize

% Convert to 3D Cartesian coordinates for surface plotting
R = AF_mag;
X = R .* sin(PH) .* cos(TH);
Y = R .* sin(PH) .* sin(TH);
Z = R .* cos(PH);

% Plot
figure;
surf(X, Y, Z, R, 'EdgeColor', 'none');
xlabel('X'); ylabel('Y'); zlabel('Z');
title(['3D Beam Pattern (Steered to Az = ' num2str(steer_az) '°, El = ' num2str(steer_el) '°)']);
colorbar;
axis equal;
view(45, 30);
