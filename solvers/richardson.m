% Solve the implicit diffusion system
%
%   (I - dtD*lap_g) X = B
%
% by preconditioned Richardson iteration, with the mean-J-scaled round-sphere
% operator M = 1 + dtD*lamJ (lamJ = lam ./ jhat) as the preconditioner.
% Splitting lap_g against the exactly invertible diagonal and moving the
% geometric part to the right-hand side gives the fixed point
%
%   X = (B + dtD*dlap(X)) ./ M
%
% iterated from the round-sphere answer. jhat is the host's minimax scale
% over the operator's symbol eigenvalues mu(x) (src/geom/geometry.ts, Jhat):
% preconditioning with lam/jhat contracts every mode and direction at rate
% (muMax - muMin)/(muMax + muMin) < 1 on any surface, where the plain lam
% (jhat = 1) diverges wherever mu > 2 — docs/reduced-transforms.md Sec 10.
% The answer never depends on jhat (the lamJ term added inside dlap is the
% term divided back out); only the convergence rate does. On the sphere
% mu = 1 and lamJ = lam.
%
% niter is fixed at compile time — the loop is unrolled into the op
% sequence, so there is no residual check and no adaptive stopping. Full
% derivation: docs/richardson-iteration.md.
%
% Written as a full re-evaluation rather than an accumulated correction on
% purpose: on the round sphere dlap is identically zero, so every iterate is
% bit for bit the first divide, with no cancellation to round differently.

function X = richardson(B, dtD, lam, filt, jhat, p2, r, dp1, dq2, jinv, niter)
  lamJ = lam ./ jhat;
  M = 1 + dtD * lamJ;
  X = B ./ M;
  for k = 1:niter
    dL = dlap(X, filt, lam, jhat, p2, r, dp1, dq2, jinv);
    X = (B + dtD * dL) ./ M;
  end
end
