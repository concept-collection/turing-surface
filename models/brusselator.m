% Brusselator reaction-diffusion on a closed surface.
%
%   du/dt = D1*lap_g(u) + A - (B+1)*u + u^2*v
%   dv/dt = D2*lap_g(v) +     B*u     - u^2*v
%
% Same scheme as models/schnakenberg.m: explicit reaction, then the implicit
% diffusion solve handed to solvers/richardson.m.

function [U, V, u, v] = init(noise, A, B)
  U = analys(A + noise);
  V = analys((B / A) * ones(numel(noise), 1));
  u = synth(U);
  v = synth(V);
end

function [Un, Vn, u, v] = step(U, V, lam, filt, gx, gy, gz, Vtx, Vty, Vtz, Vpx, Vpy, Vpz, A, B, D1, D2, dt, niter)
  u = synth(U);
  v = synth(V);
  uuv = u .* u .* v;

  Bu = U + dt * analys(A - (B + 1) * u + uuv);
  Bv = V + dt * analys(B * u - uuv);

  Un = richardson(Bu, dt * D1, lam, filt, Vtx, Vty, Vtz, Vpx, Vpy, Vpz, niter);
  Vn = richardson(Bv, dt * D2, lam, filt, Vtx, Vty, Vtz, Vpx, Vpy, Vpz, niter);
end
