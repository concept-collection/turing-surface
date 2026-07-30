% Schnakenberg reaction-diffusion on a closed surface.
%
%   du/dt = D1*lap_g(u) + a - u + u^2*v
%   dv/dt = D2*lap_g(v) + b     - u^2*v
%
% lap_g is the Laplace-Beltrami operator of the surface, which the embedding
% (gx, gy, gz) determines. The reaction is explicit on the grid; diffusion is
% implicit, which on a general surface is no longer a divide — see `step`.
%
% Provided by the caller: synth/analys (the transforms), lam = l(l+1) per
% coefficient, gx/gy/gz (the surface, on the grid) and Gx/Gy/Gz (the same
% surface as spherical-harmonic coefficients), noise (the seeded perturbation),
% niter (iterations of the implicit solve), and the parameters. Grid fields are
% npts x 1; spectral fields are real 2 x nlm (row 1 real part, row 2
% imaginary), so no complex arithmetic is needed. Each function returns the new
% spectral state followed by the grid fields to display.

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

  % Explicit reaction. Bu, Bv are the right-hand side of the implicit
  % diffusion solve  (I - dt*D*lap_g) Unew = B.
  Bu = U + dt * analys(a - u + uuv);
  Bv = V + dt * analys(b - uuv);

  % Split the surface Laplacian as  lap_g = lap_s + dlap,  where lap_s is the
  % round-sphere one. lap_s is diagonal in spherical-harmonic space with
  % eigenvalues -lam, so (I - dt*D*lap_s) inverts in a single divide and the
  % whole geometry sits inside dlap. That turns the implicit solve into
  %
  %   Unew = (Bu + dt*D1*dlap(Unew)) ./ (1 + dt*D1*lam)
  %
  % which the loop below iterates from the round-sphere answer: preconditioned
  % Richardson, with the exactly invertible round-sphere operator as the
  % preconditioner. It converges while dt*D*dlap stays small against
  % (I - dt*D*lap_s), which is what would keep the cost to a few transforms.
  Un = Bu ./ (1 + (dt * D1) * lam);
  Vn = Bv ./ (1 + (dt * D2) * lam);

  for k = 1:niter
    % ---- placeholder: dlap = lap_g - lap_s --------------------------------
    % The geometry enters HERE and nowhere else. What belongs here is the
    % Laplace-Beltrami operator of the embedding minus the round-sphere one,
    % which needs the induced metric (from derivatives of gx, gy, gz) and
    % surface derivatives of the field — transforms this project does not have
    % yet. See "The geometry is not in the operator yet" in the README.
    %
    % Until then dlap is identically zero. Not a stand-in that happens to be
    % small: it is exactly the round sphere, so this loop provably changes
    % nothing, every iterate equals the line above, and the scheme is bit for
    % bit the one turing-sphere runs. The surface is drawn, not solved on.
    dLu = 0 * Un;
    dLv = 0 * Vn;
    % ----------------------------------------------------------------------
    Un = (Bu + (dt * D1) * dLu) ./ (1 + (dt * D1) * lam);
    Vn = (Bv + (dt * D2) * dLv) ./ (1 + (dt * D2) * lam);
  end
end
