% The geometric part of the surface Laplace-Beltrami operator:
%
%   dlap = lap_g - lap_s
%
% applied to a spectral field F, where lap_g is the surface's operator and
% lap_s the round sphere's (diagonal, eigenvalues -l(l+1)). This is the piece
% of the implicit solve (I - dt*D*lap_g) X = B that a solver re-evaluates
% each iteration — the round-sphere part it inverts exactly. The operator is
% linear in F; the surface enters only through the inverse metric quantities
% Vt*/Vp* (src/geom/metric.ts), so swapping the geometry changes no code.
%
% Algorithm 3 of evolving_surface/notes/algos.tex: surface gradient of the
% field (theta/phi derivatives contracted through Vt*/Vp*); each Cartesian
% component re-analysed and differentiated again; recombined into the surface
% divergence. lam .* F adds back -lap_s(F), since lam holds +l(l+1). filt
% zeroes the top two degrees, where the theta/phi derivative recurrences
% cannot exactly represent a derivative. See docs/richardson-iteration.md.
%
% Spectral fields are real 2 x nlm; the intermediate d*/L fields live on the
% npts x 1 grid.

function dL = dlap(F, filt, Vtx, Vty, Vtz, Vpx, Vpy, Vpz, lam)
  G = F .* filt;
  Ft = dtheta(G);
  Fp = dphi(G);
  dx = Ft .* Vtx + Fp .* Vpx;
  dy = Ft .* Vty + Fp .* Vpy;
  dz = Ft .* Vtz + Fp .* Vpz;
  cx = analys(dx) .* filt;
  cy = analys(dy) .* filt;
  cz = analys(dz) .* filt;
  Ftcx = dtheta(cx);
  Fpcx = dphi(cx);
  Ftcy = dtheta(cy);
  Fpcy = dphi(cy);
  Ftcz = dtheta(cz);
  Fpcz = dphi(cz);
  L = Ftcx .* Vtx + Fpcx .* Vpx;
  L = L + Ftcy .* Vty;
  L = L + Fpcy .* Vpy;
  L = L + Ftcz .* Vtz;
  L = L + Fpcz .* Vpz;
  dL = analys(L) + lam .* F;
end
