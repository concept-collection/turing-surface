% ---------------------------------------------------------------- transforms
%
% Double-precision MATLAB ports of the operations the app provides to a .m
% around its compiled GPU pipeline. Conventions follow src/sht/layout.ts:
% orthonormal spherical harmonics with the Condon-Shortley phase, coefficients
% stored for m >= 0 only in m-major order (m = 0..mmax, l = m..lmax within
% each m) -- here as complex nlm x 1 column vectors where the GPU carries
% interleaved [re, im] pairs. Grid fields are npts x 1 columns, phi-fastest:
% point (itheta, iphi) sits at row (itheta-1)*nphi + iphi, north row first.

% Holds the precomputed tables between calls: set once from the top of the
% run, read back by every transform below.
function S = sht_tables(S)
  persistent stored
  if nargin > 0
    stored = S;
  end
  S = stored;
end

% Everything the transforms need for one grid: Gauss nodes and weights,
% per-m Legendre tables, the coefficient layout, the derivative shuffles,
% and the eigenvalue/filter vectors the models take as `lam` and `filt`.
function S = sht_setup(lmax, mmax, nlat, nphi)
  S.lmax = lmax;
  S.mmax = mmax;
  S.nlat = nlat;
  S.nphi = nphi;
  S.npts = nlat * nphi;
  [ct, wg] = gauss_legendre(nlat);
  S.ct = ct;
  S.st = sqrt(1 - ct.^2);
  S.wg = wg;
  S.nlm = (mmax + 1) * (lmax + 1) - mmax * (mmax + 1) / 2;

  % The grid angles as npts x 1 fields, phi-fastest like everything else.
  S.theta = repelem(acos(min(1, max(-1, ct))), nphi);
  S.phi = repmat(2*pi*(0:nphi-1)'/nphi, nlat, 1);
  S.stpt = repelem(S.st, nphi);

  % Degree and order of each coefficient, and each m block's start.
  off = zeros(mmax + 1, 1);
  lv = zeros(S.nlm, 1);
  mv = zeros(S.nlm, 1);
  pos = 1;
  for m = 0:mmax
    n = lmax - m + 1;
    off(m + 1) = pos;
    lv(pos:pos + n - 1) = (m:lmax)';
    mv(pos:pos + n - 1) = m;
    pos = pos + n;
  end
  S.off = off;
  S.lv = lv;
  S.mv = mv;
  % Laplace-Beltrami eigenvalues l(l+1) and the top-mode filter: 1 below
  % lmax-2, 0 at the top two degrees, where the derivative recurrences cannot
  % exactly represent a derivative (src/mgpu/model.ts).
  S.lam = lv .* (lv + 1);
  S.filt = double(lv < lmax - 2);

  % Orthonormal Legendre tables ytilde_l^m(theta_i), one nlat x (lmax-m+1)
  % block per m, by the standard three-term recurrence (src/sht/coeffs.ts;
  % SHTNS normalization, Condon-Shortley phase carried in the seed's sign).
  S.Y = cell(mmax + 1, 1);
  t = 1 / (4*pi);
  amm = sqrt(t);
  for m = 0:mmax
    if m > 0
      t = t * (2*m + 1) / (2*m);
      amm = (-1)^m * sqrt(t);
    end
    n = lmax - m + 1;
    Y = zeros(nlat, n);
    y0 = amm * S.st.^m;
    Y(:, 1) = y0;
    if n > 1
      y1 = sqrt(2*m + 3) * ct .* y0;
      Y(:, 2) = y1;
      for l = m + 2:lmax
        t1 = (l + m) * (l - m);
        a = sqrt((2*l + 1) * (2*l - 1) / t1);
        b = -sqrt(((2*l + 1) / (2*l - 3)) * ((l - 1 + m) * (l - 1 - m) / t1));
        y2 = a * ct .* y1 + b * y0;
        Y(:, l - m + 1) = y2;
        y0 = y1;
        y1 = y2;
      end
    end
    S.Y{m + 1} = Y;
  end

  % sin(theta)*dtheta in coefficient space: v_l^m = ap(lm) u_{l-1}^m +
  % am(lm) u_{l+1}^m (src/sht/derivCoeffs.ts). Neighbors sit at +-1 within
  % each m block; ap/am are zero at the block edges, so the clamped index
  % vectors never read across a boundary.
  l = lv;
  m = mv;
  ap = (l - 1) .* sqrt(max(0, (l - m) .* (l + m)) ./ ((2*l - 1) .* (2*l + 1)));
  ap(l <= m) = 0;
  am = -(l + 2) .* sqrt((l + 1 - m) .* (l + 1 + m) ./ ((2*l + 1) .* (2*l + 3)));
  am(l >= lmax) = 0;
  S.ap = ap;
  S.am = am;
  S.iprev = max((1:S.nlm)' - 1, 1);
  S.inext = min((1:S.nlm)' + 1, S.nlm);

  % dphig's Fourier multiplier: i*m on fft's frequency layout, masked past
  % the filter's reach (mcut = lmax-3), mirroring src/sht/wgsl/deriv.ts.
  freq = [(0:nphi/2)'; (-nphi/2 + 1:-1)'];
  S.dmul = 1i * freq .* (abs(freq) <= max(0, lmax - 3));
end

% Gauss-Legendre nodes cos(theta), in decreasing order (north pole first),
% and weights for integration over cos(theta) -- Newton iteration on P_n,
% as src/sht/gauss.ts.
function [x, w] = gauss_legendre(n)
  x = zeros(n, 1);
  w = zeros(n, 1);
  half = floor((n + 1) / 2);
  for i = 1:half
    z = cos(pi * (i - 0.25) / (n + 0.5));
    pp = 0;
    for it = 1:100
      p1 = 1;
      p2 = 0;
      for j = 1:n
        p3 = p2;
        p2 = p1;
        p1 = ((2*j - 1) * z * p2 - (j - 1) * p3) / j;
      end
      pp = n * (z * p1 - p2) / (z^2 - 1);
      dz = p1 / pp;
      z = z - dz;
      if abs(dz) < 1e-15 * abs(z) + 1e-300
        p1 = 1;
        p2 = 0;
        for j = 1:n
          p3 = p2;
          p2 = p1;
          p1 = ((2*j - 1) * z * p2 - (j - 1) * p3) / j;
        end
        pp = n * (z * p1 - p2) / (z^2 - 1);
        z = z - p1 / pp;
        break;
      end
    end
    x(i) = z;
    x(n + 1 - i) = -z;
    wi = 2 / ((1 - z^2) * pp^2);
    w(i) = wi;
    w(n + 1 - i) = wi;
  end
  if mod(n, 2) == 1
    x(half) = 0;
  end
end

% Synthesis, spectral -> grid. Grouped calls -- [a, b] = synth(x, y) -- are
% the app's batching hint; here each member simply runs in turn.
function varargout = synth(varargin)
  S = sht_tables();
  varargout = cell(1, nargin);
  for k = 1:nargin
    varargout{k} = synth_one(S, varargin{k});
  end
end

function f = synth_one(S, Q)
  % Legendre stage per m, then one inverse FFT per latitude ring with the
  % m < 0 modes filled in by conjugate symmetry (the field is real).
  G = zeros(S.nphi, S.nlat);
  for m = 0:S.mmax
    Fm = (S.Y{m + 1} * Q(S.off(m + 1):S.off(m + 1) + S.lmax - m)).';
    G(m + 1, :) = Fm;
    if m > 0
      G(S.nphi + 1 - m, :) = conj(Fm);
    end
  end
  f = S.nphi * real(ifft(G, [], 1));
  f = f(:);
end

% Analysis, grid -> spectral: forward FFT per ring, then Gauss quadrature
% against the same Legendre tables.
function varargout = analys(varargin)
  S = sht_tables();
  varargout = cell(1, nargin);
  for k = 1:nargin
    varargout{k} = analys_one(S, varargin{k});
  end
end

function Q = analys_one(S, f)
  F = fft(reshape(f, S.nphi, S.nlat), [], 1) * (2*pi/S.nphi);
  Q = complex(zeros(S.nlm, 1));
  for m = 0:S.mmax
    Q(S.off(m + 1):S.off(m + 1) + S.lmax - m) = S.Y{m + 1}.' * (S.wg .* F(m + 1, :).');
  end
end

% The coefficients of sin(theta)*dtheta(u): the alpha^+/alpha^- shift by one
% degree within each m block.
function V = dthetac(Q)
  S = sht_tables();
  V = S.ap .* Q(S.iprev) + S.am .* Q(S.inext);
end

% The coefficients of dphi(u): i*m, diagonal.
function V = dphic(Q)
  S = sht_tables();
  V = 1i * (S.mv .* Q);
end

% Grid-space derivatives, coefficients in: compositions of the shuffles and
% the synthesis (src/sht/deriv.ts). dtheta divides by sin(theta) afterwards.
function f = dtheta(Q)
  S = sht_tables();
  f = synth(dthetac(Q)) ./ S.stpt;
end

function f = dphi(Q)
  f = synth(dphic(Q));
end

% Grid-space phi derivative, grid in: two FFT stages and a pointwise i*m,
% no Legendre work -- d/dphi is diagonal in the Fourier index.
function g = dphig(f)
  S = sht_tables();
  F = fft(reshape(f, S.nphi, S.nlat), [], 1);
  g = real(ifft(S.dmul .* F, [], 1));
  g = g(:);
end

% ---------------------------------------------------------------- the surface
%
% What the app precomputes from a shape's raw grid values: the band-limited
% embedding and both metric formulations built on it (src/geom/geometry.ts,
% src/geom/metric.ts). The solver runs on the synthesis of the coefficients,
% not on the raw values -- for a shape with sharp features the two differ.
function G = surface_tables(gxr, gyr, gzr)
  S = sht_tables();
  [G.Gx, G.Gy, G.Gz] = analys(gxr, gyr, gzr);
  [G.gx, G.gy, G.gz] = synth(G.Gx, G.Gy, G.Gz);

  % Flux-form metric weights, from the sin-weighted theta tangent
  % sin(theta)*X_theta and X_phi, both smooth on the sphere:
  % gtt~ = sin^2 g_tt, gtp~ = sin g_tp, D = J sin^2(theta).
  [sXtx, sXty, sXtz] = synth(dthetac(G.Gx), dthetac(G.Gy), dthetac(G.Gz));
  [Xpx, Xpy, Xpz] = synth(dphic(G.Gx), dphic(G.Gy), dphic(G.Gz));
  gtt = sXtx.^2 + sXty.^2 + sXtz.^2;
  gtp = sXtx.*Xpx + sXty.*Xpy + sXtz.*Xpz;
  gpp = Xpx.^2 + Xpy.^2 + Xpz.^2;
  D = sqrt(gtt .* gpp - gtp.^2);
  G.p1 = gpp ./ D;
  G.p2 = -gtp ./ D;
  G.q2 = gtt ./ D;
  G.r = 1 ./ D;

  % The sphere-subtracted weights and the bounded 1/J = r sin^2(theta) --
  % what keeps the concentrated division off the round sphere's share of the
  % flux divergence. Formed here in f64, as the app forms them.
  G.jinv = G.r .* S.stpt.^2;
  G.dp1 = G.p1 - 1;
  G.dq2 = G.q2 - 1;

  % Preconditioner scale Jhat = 2/(muMin + muMax) over the eigenvalues of
  % the operator's symbol S = (1/J)[[p1, p2], [p2, q2]].
  s11 = G.p1 .* G.jinv;
  s12 = G.p2 .* G.jinv;
  s22 = G.q2 .* G.jinv;
  mn = (s11 + s22) / 2;
  disc = sqrt(((s11 - s22) / 2).^2 + s12.^2);
  G.Jhat = 2 / (min(mn - disc) + max(mn + disc));

  % Inverse metric quantities V_theta/V_phi, for the Algorithm-4 models.
  Xtx = sXtx ./ S.stpt;
  Xty = sXty ./ S.stpt;
  Xtz = sXtz ./ S.stpt;
  g11 = Xtx.^2 + Xty.^2 + Xtz.^2;
  g12 = Xtx.*Xpx + Xty.*Xpy + Xtz.*Xpz;
  g22 = gpp;
  det = g11 .* g22 - g12.^2;
  G.Vtx = (g22 .* Xtx - g12 .* Xpx) ./ det;
  G.Vty = (g22 .* Xty - g12 .* Xpy) ./ det;
  G.Vtz = (g22 .* Xtz - g12 .* Xpz) ./ det;
  G.Vpx = (g11 .* Xpx - g12 .* Xtx) ./ det;
  G.Vpy = (g11 .* Xpy - g12 .* Xty) ./ det;
  G.Vpz = (g11 .* Xpz - g12 .* Xtz) ./ det;
end

% ---------------------------------------------------------------- random field
%
% chebfun-style smooth random field in 3D, restricted to the surface by
% evaluating it at the grid points -- the way surfacefun seeds a run. Two
% signatures, as in the app:
%   [k, c] = randnfun3(lambda, dom)     the Fourier-mode draw (tools/randnfun3.m)
%   f = randnfun3(lambda, gx, gy, gz)   that draw, summed at the surface points
% Seed with rng(...) before calling.
function varargout = randnfun3(lambda, varargin)
  if nargin == 2
    [k, c] = randnfun3_modes(lambda, varargin{1});
    varargout = {k, c};
    return;
  end
  [gx, gy, gz] = deal(varargin{1:3});
  dom = [min(gx) max(gx) min(gy) max(gy) min(gz) max(gz)];
  [k, c] = randnfun3_modes(lambda, dom);
  % Summed in blocks of modes: the full npts x nmodes phase matrix can reach
  % hundreds of MB at a fine wavelength.
  f = zeros(numel(gx), 1);
  blk = 2048;
  for j0 = 1:blk:size(k, 1)
    j1 = min(j0 + blk - 1, size(k, 1));
    t = gx * k(j0:j1, 1)' + gy * k(j0:j1, 2)' + gz * k(j0:j1, 3)';
    f = f + cos(t) * c(j0:j1, 1) - sin(t) * c(j0:j1, 2);
  end
  varargout = {f};
end

% ---------------------------------------------------------------- display
%
% The pattern on the surface, one panel per species. The solver grid has no
% pole rows and an open phi seam; wrap_grid closes both for display, capping
% each pole with the mean of its nearest ring.
function h = plot_setup(gx, gy, gz, fields, names)
  fig = figure('Name', 'turing-surface', 'Color', 'w');
  Xs = wrap_grid(gx);
  Ys = wrap_grid(gy);
  Zs = wrap_grid(gz);
  n = numel(fields);
  h.surf = gobjects(1, n);
  h.ax = gobjects(1, n);
  for k = 1:n
    ax = subplot(1, n, k, 'Parent', fig);
    h.surf(k) = surf(ax, Xs, Ys, Zs, wrap_grid(fields{k}), 'EdgeColor', 'none');
    shading(ax, 'interp');
    axis(ax, 'equal');
    axis(ax, 'off');
    colormap(ax, 'jet');
    colorbar(ax);
    h.ax(k) = ax;
  end
  h.names = names;
end

function plot_update(h, fields, t, k, nsteps)
  for i = 1:numel(fields)
    C = wrap_grid(fields{i});
    set(h.surf(i), 'CData', C);
    lo = min(C(:));
    hi = max(C(:));
    if ~(hi > lo)
      hi = lo + 1;
    end
    caxis(h.ax(i), [lo hi]);
    title(h.ax(i), sprintf('%s   t = %.3f   (step %d/%d)', h.names{i}, t, k, nsteps));
  end
  drawnow;
end

function M = wrap_grid(f)
  S = sht_tables();
  M = reshape(f, S.nphi, S.nlat).';
  M = [M, M(:, 1)];
  M = [mean(M(1, :)) * ones(1, S.nphi + 1); M; mean(M(end, :)) * ones(1, S.nphi + 1)];
end

% ---------------------------------------------------------------- results file
%
% Complex coefficients -> flat float32 [re, im] per (l, m), the layout the
% app's reference-file reader expects (docs/ellipsoid-reference-spec.md).
function write_coeffs(fname, path, Q)
  flat = zeros(2 * numel(Q), 1);
  flat(1:2:end) = real(Q);
  flat(2:2:end) = imag(Q);
  h5create(fname, path, numel(flat), 'Datatype', 'single');
  h5write(fname, path, single(flat));
end

% h5writeatt cannot create a bare group, so the attribute-only groups of the
% reference layout are made through the low-level API.
function make_group(fname, path)
  fid = H5F.open(fname, 'H5F_ACC_RDWR', 'H5P_DEFAULT');
  gid = H5G.create(fid, path, 'H5P_DEFAULT', 'H5P_DEFAULT', 'H5P_DEFAULT');
  H5G.close(gid);
  H5F.close(fid);
end
