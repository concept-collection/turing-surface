% Allen-Cahn on a closed surface: interfaces form, then coarsen.
%
%   du/dt = eps2*lap_g(u) + u - u^3
%
% Same scheme as models/schnakenberg.m.

function [U, u] = init(noise)
  U = analys(noise);
  u = synth(U);
end

function [Un, u] = step(U, lam, filt, gx, gy, gz, Vtx, Vty, Vtz, Vpx, Vpy, Vpz, eps2, dt, niter)
  u = synth(U);

  Bu = U + dt * analys(u - u.^3);
  Un = Bu ./ (1 + (dt * eps2) * lam);

  for k = 1:niter
    % dlap = lap_g - lap_s, evaluated at the current iterate (see
    % models/schnakenberg.m and docs/richardson-iteration.md for the
    % derivation).
    Fu = Un .* filt;
    Ftu = dtheta(Fu);
    Fpu = dphi(Fu);
    dux = Ftu .* Vtx + Fpu .* Vpx;
    duy = Ftu .* Vty + Fpu .* Vpy;
    duz = Ftu .* Vtz + Fpu .* Vpz;
    cux = analys(dux) .* filt;
    cuy = analys(duy) .* filt;
    cuz = analys(duz) .* filt;
    Ftcux = dtheta(cux);
    Fpcux = dphi(cux);
    Ftcuy = dtheta(cuy);
    Fpcuy = dphi(cuy);
    Ftcuz = dtheta(cuz);
    Fpcuz = dphi(cuz);
    lapu = Ftcux .* Vtx + Fpcux .* Vpx;
    lapu = lapu + Ftcuy .* Vty;
    lapu = lapu + Fpcuy .* Vpy;
    lapu = lapu + Ftcuz .* Vtz;
    lapu = lapu + Fpcuz .* Vpz;
    dLu = analys(lapu) + lam .* Un;

    Un = (Bu + (dt * eps2) * dLu) ./ (1 + (dt * eps2) * lam);
  end
end
