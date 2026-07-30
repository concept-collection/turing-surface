% Brusselator reaction-diffusion on a closed surface. Turing stripes and spots,
% from a smaller diffusivity contrast than Schnakenberg but a stiffer reaction.
%
%   du/dt = D1*lap_g(u) + A - (B+1)*u + u^2*v
%   dv/dt = D2*lap_g(v) +     B*u     - u^2*v
%
% See models/schnakenberg.m for what the caller provides and for how the
% implicit solve is split between the round-sphere operator and the geometry.

function [U, V, u, v] = init(noise, A, B)
  U = analys(A + noise);
  V = analys((B / A) * ones(numel(noise), 1));
  u = synth(U);
  v = synth(V);
end

function [Un, Vn, u, v] = step(U, V, lam, gx, gy, gz, A, B, D1, D2, dt, niter)
  u = synth(U);
  v = synth(V);
  uuv = u .* u .* v;

  Bu = U + dt * analys(A - (B + 1) * u + uuv);
  Bv = V + dt * analys(B * u - uuv);

  Un = Bu ./ (1 + (dt * D1) * lam);
  Vn = Bv ./ (1 + (dt * D2) * lam);

  for k = 1:niter
    % ---- placeholder: dlap = lap_g - lap_s (see models/schnakenberg.m) ----
    dLu = 0 * Un;
    dLv = 0 * Vn;
    % ----------------------------------------------------------------------
    Un = (Bu + (dt * D1) * dLu) ./ (1 + (dt * D1) * lam);
    Vn = (Bv + (dt * D2) * dLv) ./ (1 + (dt * D2) * lam);
  end
end
