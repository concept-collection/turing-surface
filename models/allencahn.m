% Allen-Cahn on a closed surface: interfaces form, then coarsen.
%
%   du/dt = eps2*lap_g(u) + u - u^3
%
% Same scheme as models/schnakenberg.m: the step assembles the right-hand
% side and hands the implicit diffusion solve to solve(...) — the solver the
% app's selector picks — with the operator's geometric part in lib/dlap.m.

% Seeded from a smooth random field -- see models/schnakenberg.m.
function [U, u] = init(lam3, gx, gy, gz)
  U = analys(0.01 * randnfun3(lam3, gx, gy, gz));
  u = synth(U);
end

function [Un, u] = step(U, lam, filt, wlm, gx, gy, gz, p2, r, dp1, dq2, jinv, jhat, eps2, dt, nlm, niter)
  u = synth(U);

  Bu = U + dt * analys(u - u.^3);

  Un = solve(Bu, dt * eps2, lam, filt, wlm, jhat, p2, r, dp1, dq2, jinv, nlm, niter);
end
