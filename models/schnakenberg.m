% Schnakenberg reaction-diffusion on a closed surface.
%
%   du/dt = D1*lap_g(u) + a - u + u^2*v
%   dv/dt = D2*lap_g(v) + b     - u^2*v
%
% Explicit reaction, implicit diffusion (IMEX Euler): the step forms the
% right-hand side B of the linear system (I - dt*D*lap_g) Unew = B, and hands
% the solve to solve(...) — the solver the app's selector picks (richardson,
% bicgstab or gmres, see solvers/), every one applying the operator's
% geometric part through lib/dlap.m: the flux-form evaluation with the
% sphere-split divergence and the mean-J preconditioner (jhat), 6 transforms
% per species per solve iteration where the Cartesian-gradient form
% (models/schnakenberg_alg4.m, kept as a live reference) needs 12. A model
% may also name a solver directly in these two call lines. Grid fields are
% npts x 1; spectral fields are real 2 x nlm. See
% docs/richardson-iteration.md and docs/reduced-transforms.md.

% The uniform steady state, perturbed by a smooth random field: chebfun's
% randnfun3 on the surface's bounding box, restricted to the surface by
% evaluating it at the grid points -- the way surfacefun seeds a run. lam3
% is its wavelength; the draw is seeded on the host, the sum over its
% Fourier modes runs on the GPU (src/mgpu/randnfun3.ts).
function [U, V, u, v] = init(lam3, gx, gy, gz, a, b)
  f = randnfun3(lam3, gx, gy, gz);
  us = a + b;
  vs = b / (us * us);
  [U, V] = analys(us + 0.01*f, vs * ones(numel(f), 1));
  [u, v] = synth(U, V);
end

function [Un, Vn, u, v] = step(U, V, lam, filt, wlm, gx, gy, gz, p2, r, dp1, dq2, jinv, jhat, a, b, D1, D2, dt, nlm, niter)
  % Grouped transforms -- [a, b] = synth(x, y) -- are explicit batching:
  % output k is the transform of input k, and the whole group runs as one
  % batched Legendre dispatch, or as many as the device's lane width allows
  % (src/mgpu/plan.ts, materializeTransforms). The grouping is a promise of
  % independence, never of a lane width, so the same source runs anywhere.
  [u, v] = synth(U, V);
  uuv = u .* u .* v;

  % Right-hand side of the implicit solve (I - dt*D*lap_g) Unew = B.
  ru = a - u + uuv;
  rv = b - uuv;
  [Ru, Rv] = analys(ru, rv);
  Bu = U + dt * Ru;
  Bv = V + dt * Rv;

  % The species diffuse independently, so each gets its own solve.
  Un = solve(Bu, dt * D1, lam, filt, wlm, jhat, p2, r, dp1, dq2, jinv, nlm, niter);
  Vn = solve(Bv, dt * D2, lam, filt, wlm, jhat, p2, r, dp1, dq2, jinv, nlm, niter);
end
