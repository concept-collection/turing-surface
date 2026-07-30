% Allen-Cahn on a closed surface. One species: interfaces form and then coarsen
% until one domain swallows the surface.
%
%   du/dt = eps2*lap_g(u) + u - u^3
%
% See models/schnakenberg.m for what the caller provides and for how the
% implicit solve is split between the round-sphere operator and the geometry.

function [U, u] = init(noise)
  U = analys(noise);
  u = synth(U);
end

function [Un, u] = step(U, lam, gx, gy, gz, eps2, dt, niter)
  u = synth(U);

  Bu = U + dt * analys(u - u.^3);
  Un = Bu ./ (1 + (dt * eps2) * lam);

  for k = 1:niter
    % ---- placeholder: dlap = lap_g - lap_s (see models/schnakenberg.m) ----
    dLu = 0 * Un;
    % ----------------------------------------------------------------------
    Un = (Bu + (dt * eps2) * dLu) ./ (1 + (dt * eps2) * lam);
  end
end
