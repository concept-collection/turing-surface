% Schnakenberg reaction-diffusion on a closed surface.
%
%   du/dt = D1*lap_g(u) + a - u + u^2*v
%   dv/dt = D2*lap_g(v) + b     - u^2*v
%
% Explicit reaction, implicit diffusion (IMEX Euler): the step forms the
% right-hand side B of the linear system (I - dt*D*lap_g) Unew = B, and hands
% the solve to solve(...) — the solver the app's selector picks (richardson,
% bicgstab or gmres, see solvers/), every one applying the operator's
% geometric part through lib/dlap.m. A model may also name a solver directly
% in these two call lines. Grid fields are npts x 1; spectral fields are
% real 2 x nlm. See docs/richardson-iteration.md.

function [U, V, u, v] = init(noise, a, b)
  us = a + b;
  vs = b / (us * us);
  U = analys(us + noise);
  V = analys(vs * ones(numel(noise), 1));
  u = synth(U);
  v = synth(V);
end

function [Un, Vn, u, v] = step(U, V, lam, filt, wlm, gx, gy, gz, Vtx, Vty, Vtz, Vpx, Vpy, Vpz, a, b, D1, D2, dt, nlm, niter)
  u = synth(U);
  v = synth(V);
  uuv = u .* u .* v;

  % Right-hand side of the implicit solve (I - dt*D*lap_g) Unew = B.
  Bu = U + dt * analys(a - u + uuv);
  Bv = V + dt * analys(b - uuv);

  % The species diffuse independently, so each gets its own solve.
  Un = solve(Bu, dt * D1, lam, filt, wlm, Vtx, Vty, Vtz, Vpx, Vpy, Vpz, nlm, niter);
  Vn = solve(Bv, dt * D2, lam, filt, wlm, Vtx, Vty, Vtz, Vpx, Vpy, Vpz, nlm, niter);
end
