% Solve the implicit diffusion system
%
%   (I - dtD*lap_g) X = B,  i.e.  A*X = B  with  A*x = M.*x - dtD*dlap(x)
%
% by preconditioned BiCGSTAB (van der Vorst), with the round-sphere operator
% M = 1 + dtD*lam as the preconditioner — the same M solvers/richardson.m
% inverts, applied here as M\v inside the Krylov recurrence. Same operator,
% different solver: where Richardson converges only while the geometric
% correction stays small against M, BiCGSTAB builds a Krylov space and
% converges on configurations well outside that radius, at the price of two
% dlap evaluations per iteration instead of one, plus three dot products.
%
% The scalars (rho, alpha, omega) are GPU-resident 1-element values: `dot` is
% a reduction dispatch and the recurrences on its results compile to
% 1-element kernels, so the whole solve is still one command stream with no
% CPU in the loop. Inner products are taken in the half-spectrum weighting
% wlm (m > 0 counts twice), so they are the real L2 inner products on the
% sphere; the weight is folded into the fixed shadow residual once, and into
% t per iteration.
%
% niter is fixed at compile time: no residual test, and so no early exit —
% which for BiCGSTAB matters at the *converged* end, where the residual is
% fp32 noise and every textbook ratio is 0/0. Each ratio a/b is therefore
% written in the guarded form a*b/(b*b + 1e-30): identical to a/b (to a
% couple of ulp) whenever b is meaningfully sized, and 0 when it is not, so
% a converged or broken-down iteration degrades to a stationary one
% (coefficients 0, X carried unchanged) instead of poisoning the state with
% NaNs. X0 is the round-sphere answer, so niter = 0 computes richardson's
% starting divide.

function X = bicgstab(B, dtD, lam, filt, wlm, Vtx, Vty, Vtz, Vpx, Vpy, Vpz, niter)
  M = 1 + dtD * lam;
  X = B ./ M;
  dL0 = dlap(X, filt, Vtx, Vty, Vtz, Vpx, Vpy, Vpz, lam);
  R = B - (M .* X - dtD * dL0);
  Rh = R .* wlm;
  P = 0 * R;
  V = 0 * R;
  rho = 1;
  alpha = 1;
  omega = 1;
  for k = 1:niter
    rho1 = dot(Rh, R);
    beta = (rho1 * rho) / (rho * rho + 1e-30) * ((alpha * omega) / (omega * omega + 1e-30));
    P = R + beta * (P - omega * V);
    Ph = P ./ M;
    dLp = dlap(Ph, filt, Vtx, Vty, Vtz, Vpx, Vpy, Vpz, lam);
    V = M .* Ph - dtD * dLp;
    rv = dot(Rh, V);
    alpha = (rho1 * rv) / (rv * rv + 1e-30);
    S = R - alpha * V;
    Sh = S ./ M;
    dLs = dlap(Sh, filt, Vtx, Vty, Vtz, Vpx, Vpy, Vpz, lam);
    T = M .* Sh - dtD * dLs;
    Tw = T .* wlm;
    ts = dot(Tw, S);
    tt = dot(Tw, T);
    omega = (ts * tt) / (tt * tt + 1e-30);
    X = X + alpha * Ph + omega * Sh;
    R = S - omega * T;
    rho = rho1;
  end
end
