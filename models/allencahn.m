% Allen-Cahn on a closed surface: interfaces form, then coarsen.
%
%   du/dt = eps2*lap_g(u) + u - u^3
%
% Same scheme as models/schnakenberg.m: explicit reaction, then the implicit
% diffusion solve handed to solvers/richardson.m.

function [U, u] = init(noise)
  U = analys(noise);
  u = synth(U);
end

function [Un, u] = step(U, lam, filt, gx, gy, gz, Vtx, Vty, Vtz, Vpx, Vpy, Vpz, eps2, dt, niter)
  u = synth(U);

  Bu = U + dt * analys(u - u.^3);
  Un = richardson(Bu, dt * eps2, lam, filt, Vtx, Vty, Vtz, Vpx, Vpy, Vpz, niter);
end
