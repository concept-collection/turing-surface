% Smooth random function in 3D — chebfun's randnfun3, as the Fourier modes
% it is built from rather than as a chebfun3.
%
%   [K, C] = randnfun3(LAMBDA, DOM) draws a random trig series on the box
%   DOM = [x0 x1 y0 y1 z0 z1] with maximum frequency about 2*pi/LAMBDA in
%   each direction and standard normal distribution N(0,1) at each point.
%   K is nmodes x 3 (angular wavenumbers) and C is nmodes x 2 (real and
%   imaginary parts), defining
%
%       f(x,y,z) = sum_j  C(j,1)*cos(K(j,:)*[x;y;z]) - C(j,2)*sin(K(j,:)*[x;y;z])
%
% Seed the draw with rng(...) before calling.
%
% chebfun returns a chebfun3 and evaluates it later; this project has no
% such object, and the sum above is what the GPU evaluates at the surface
% points (src/mgpu/randnfun3.ts). Splitting it here is also what keeps the
% draw in MATLAB: randn has no counterpart in the compiled WGSL dialect.

function [k, c] = randnfun3(lambda, dom)
  % chebfun's nonperiodic path builds a periodic function on a domain about
  % 20% larger and restricts it. Restriction is free when evaluating at
  % points, so we keep the enlarged period and never form the smaller one.
  m = round(1.2*(dom(2)-dom(1))/lambda + 2);
  n = round(1.2*(dom(4)-dom(3))/lambda + 2);
  p = round(1.2*(dom(6)-dom(5))/lambda + 2);
  m2 = 2*m+1;
  n2 = 2*n+1;
  p2 = 2*p+1;
  N = m2*n2*p2;

  % chebfun draws the whole cube (column-major) before masking; drawing in
  % that same order keeps a seed meaning the same thing here as there.
  cr = randn(N, 1);
  ci = randn(N, 1);

  % The cube's integer wavenumbers, -m:m x -n:n x -p:p in column-major order.
  i = (0:N-1).';
  jx = mod(i, m2) - m;
  jy = mod(floor(i/m2), n2) - n;
  jz = floor(i/(m2*n2)) - p;

  % Confine to a ball for isotropy.
  keep = ((jx/m).^2 + (jy/n).^2 + (jz/p).^2) <= 1;
  jx = jx(keep);
  jy = jy(keep);
  jz = jz(keep);
  cr = cr(keep);
  ci = ci(keep);

  % Normalize so the variance is 1 at each point.
  s = 1/sqrt(numel(cr));
  cr = s*cr;
  ci = s*ci;

  % Angular wavenumbers on the enlarged period, which is a whole number of
  % wavelengths on each side.
  kx = 2*pi*jx/(m*lambda);
  ky = 2*pi*jy/(n*lambda);
  kz = 2*pi*jz/(p*lambda);

  % Fold the box's origin into the phase, so evaluating is a plain sum over
  % cos(k.x) and sin(k.x) with no offset left to carry.
  ph = -(kx*dom(1) + ky*dom(3) + kz*dom(5));
  k = [kx, ky, kz];
  c = [cr.*cos(ph) - ci.*sin(ph), cr.*sin(ph) + ci.*cos(ph)];
end
