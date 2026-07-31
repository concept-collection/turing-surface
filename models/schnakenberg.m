% Schnakenberg reaction-diffusion on a closed surface.
%
%   du/dt = D1*lap_g(u) + a - u + u^2*v
%   dv/dt = D2*lap_g(v) + b     - u^2*v
%
% Explicit reaction, implicit diffusion (IMEX Euler). The implicit solve
% splits lap_g = lap_s + dlap: the round-sphere part lap_s is diagonal in
% spherical-harmonic space (eigenvalues -lam), and the loop iterates the
% geometric correction dlap from that exact solve. Grid fields are npts x 1;
% spectral fields are real 2 x nlm.

function [U, V, u, v] = init(noise, a, b)
  us = a + b;
  vs = b / (us * us);
  U = analys(us + noise);
  V = analys(vs * ones(numel(noise), 1));
  u = synth(U);
  v = synth(V);
end

function [Un, Vn, u, v] = step(U, V, lam, gx, gy, gz, a, b, D1, D2, dt, niter)
  u = synth(U);
  v = synth(V);
  uuv = u .* u .* v;

  % Right-hand side of the implicit solve (I - dt*D*lap_g) Unew = B.
  Bu = U + dt * analys(a - u + uuv);
  Bv = V + dt * analys(b - uuv);

  % Round-sphere solve, then iterate the geometric correction.
  Un = Bu ./ (1 + (dt * D1) * lam);
  Vn = Bv ./ (1 + (dt * D2) * lam);

  for k = 1:niter
    % Placeholder: dlap = lap_g - lap_s is still identically zero, so this
    % is exactly the round-sphere solver and the geometry is only drawn.
    % See the README.
    dLu = 0 * Un;
    dLv = 0 * Vn;
    Un = (Bu + (dt * D1) * dLu) ./ (1 + (dt * D1) * lam);
    Vn = (Bv + (dt * D2) * dLv) ./ (1 + (dt * D2) * lam);
  end
end
