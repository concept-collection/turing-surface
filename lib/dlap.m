% The geometric part of the surface Laplace-Beltrami operator, plus the
% preconditioner's round-sphere share:
%
%   dlap(F) = (analys(lap_g F) + lamJ .* F) .* filt
%
% applied to a spectral field F, where lap_g is the surface's operator and
% lamJ = lam ./ jhat is the mean-J preconditioner's diagonal (lam holds
% +l(l+1), so lamJ .* F adds back -lap_s(F)/jhat exactly). This is the piece
% of the implicit solve (I - dt*D*lap_g) X = B that a solver re-evaluates
% each iteration — the diagonal part M = 1 + dtD*lamJ it inverts exactly.
% The operator is linear in F; the surface enters only through the
% precomputed flux weights (src/geom/metric.ts), so swapping the geometry
% changes no code.
%
% lap_g is evaluated in flux form (docs/reduced-transforms.md): the
% sin-weighted derivatives Ft = sin(theta)*dtheta(F) and Fp = dphi(F) — both
% smooth on the sphere, synthesized straight from the dthetac/dphic
% coefficient shuffles — are combined pointwise through the precomputed
% weights into two fluxes P,Q, also smooth. The theta flux P goes back to
% coefficients, through the same shuffle again, and is synthesized as
% sin(theta)*dtheta(P); the phi flux Q never leaves the grid — d/dphi is
% diagonal in the Fourier index, so dphig differentiates it with two FFT
% stages and no Legendre work. Their sum, scaled by r, is the deviation's
% divergence; the only division by sin(theta) anywhere is folded into the
% weights at precompute time.
%
% The weights here are the *sphere-subtracted* ones: dp1 = p1 - 1 and
% dq2 = q2 - 1 (p2 is zero on the sphere already), so P,Q are the deviation
% fluxes. What that leaves out is the round sphere's own divergence,
% -sin^2(theta)*lap_s(F), which needs no flux machinery at all: lap_s is
% diagonal, so it is -lam.*G synthesized once (S below, riding in the
% gradient's batched synthesis) and scaled by the bounded jinv = 1/J =
% r*sin^2(theta). r therefore multiplies only the deviation — the difference
% between this and multiplying the whole flux is two orders of magnitude of
% polar roundoff, and it is what keeps a pattern from nucleating at the pole
% (docs/reduced-transforms.md Sec 5).
%
% filt zeroes the top two degrees, where the derivative recurrences cannot
% exactly represent a derivative, and the result is projected onto the same
% band (algos.tex Algorithm 5 zeroes the same coefficients): without that,
% each solve iteration replaces a bit more of the top degrees' implicit
% diffusion with nothing, and multi-species systems un-diffuse at different
% rates — a spurious Turing band at the band edge.
%
% Cost: 6 Legendre transforms (a 3-wide synthesis batch, one analysis, one
% 1-wide synthesis, one final analysis) plus the FFT-only dphig and two
% coefficient shuffles. Spectral fields are real 2 x nlm; the intermediate
% fields live on the npts x 1 grid.

function dL = dlap(F, filt, lam, jhat, p2, r, dp1, dq2, jinv)
  lamJ = lam ./ jhat;
  G = F .* filt;
  vt = dthetac(G);
  vp = dphic(G);
  S0 = lam .* G;
  [Ft, Fp, S] = synth(vt, vp, S0);
  P = dp1 .* Ft + p2 .* Fp;
  Q = p2 .* Ft + dq2 .* Fp;
  PA = analys(P);
  Pc = PA .* filt;
  sc = dthetac(Pc);
  L = synth(sc);
  dQ = dphig(Q);
  lapF = r .* (L + dQ) - jinv .* S;
  LA = analys(lapF);
  dL = (LA + lamJ .* F) .* filt;
end
