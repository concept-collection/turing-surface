% Solve the implicit diffusion system
%
%   (I - dtD*lap_g) X = B
%
% by preconditioned Richardson iteration, with the round-sphere operator as
% the preconditioner. Splitting lap_g = lap_s + dlap and moving the geometric
% part to the right-hand side gives the fixed point
%
%   X = (B + dtD*dlap(X)) ./ (1 + dtD*lam)
%
% iterated from the round-sphere answer (lam holds +l(l+1), so the divide is
% the exact inverse of I - dtD*lap_s). It converges while dtD*dlap stays
% small against what the divide already inverts; niter is fixed at compile
% time — the loop is unrolled into the op sequence, so there is no residual
% check and no adaptive stopping. Full derivation and the map onto
% algos.tex's GMRES formulation: docs/richardson-iteration.md.
%
% Written as a full re-evaluation rather than an accumulated correction on
% purpose: on the round sphere dlap is identically zero, so every iterate is
% bit for bit the first divide, with no cancellation to round differently.

function X = richardson(B, dtD, lam, filt, Vtx, Vty, Vtz, Vpx, Vpy, Vpz, niter)
  X = B ./ (1 + dtD * lam);
  for k = 1:niter
    dL = dlap(X, filt, Vtx, Vty, Vtz, Vpx, Vpy, Vpz, lam);
    X = (B + dtD * dL) ./ (1 + dtD * lam);
  end
end
