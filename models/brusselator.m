% Brusselator reaction-diffusion on a closed surface.
%
%   du/dt = D1*lap_g(u) + A - (B+1)*u + u^2*v
%   dv/dt = D2*lap_g(v) +     B*u     - u^2*v
%
% Same scheme as models/schnakenberg.m: the step assembles the right-hand
% side and hands the implicit diffusion solve to solve(...) — the solver the
% app's selector picks — with the operator's geometric part in lib/dlap.m.

% Seeded from a smooth random field -- see models/schnakenberg.m.
function [U, V, u, v] = init(lam3, gx, gy, gz, A, B)
  f = randnfun3(lam3, gx, gy, gz);
  [U, V] = analys(A + 0.01*f, (B / A) * ones(numel(f), 1));
  [u, v] = synth(U, V);
end

function [Un, Vn, u, v] = step(U, V, lam, filt, wlm, gx, gy, gz, p2, r, dp1, dq2, jinv, jhat, A, B, D1, D2, dt, nlm, niter)
  [u, v] = synth(U, V);
  uuv = u .* u .* v;

  ru = A - (B + 1) * u + uuv;
  rv = B * u - uuv;
  [Ru, Rv] = analys(ru, rv);
  Bu = U + dt * Ru;
  Bv = V + dt * Rv;

  Un = solve(Bu, dt * D1, lam, filt, wlm, jhat, p2, r, dp1, dq2, jinv, nlm, niter);
  Vn = solve(Bv, dt * D2, lam, filt, wlm, jhat, p2, r, dp1, dq2, jinv, nlm, niter);
end
