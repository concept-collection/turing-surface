% Brusselator reaction-diffusion on a closed surface.
%
%   du/dt = D1*lap_g(u) + A - (B+1)*u + u^2*v
%   dv/dt = D2*lap_g(v) +     B*u     - u^2*v
%
% Same scheme as models/schnakenberg.m.

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

  Un = Bu ./ (1 + (dt * D1) * lam);
  Vn = Bv ./ (1 + (dt * D2) * lam);

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

    Fv = Vn .* filt;
    Ftv = dtheta(Fv);
    Fpv = dphi(Fv);
    dvx = Ftv .* Vtx + Fpv .* Vpx;
    dvy = Ftv .* Vty + Fpv .* Vpy;
    dvz = Ftv .* Vtz + Fpv .* Vpz;
    cvx = analys(dvx) .* filt;
    cvy = analys(dvy) .* filt;
    cvz = analys(dvz) .* filt;
    Ftcvx = dtheta(cvx);
    Fpcvx = dphi(cvx);
    Ftcvy = dtheta(cvy);
    Fpcvy = dphi(cvy);
    Ftcvz = dtheta(cvz);
    Fpcvz = dphi(cvz);
    lapv = Ftcvx .* Vtx + Fpcvx .* Vpx;
    lapv = lapv + Ftcvy .* Vty;
    lapv = lapv + Fpcvy .* Vpy;
    lapv = lapv + Ftcvz .* Vtz;
    lapv = lapv + Fpcvz .* Vpz;
    dLv = analys(lapv) + lam .* Vn;

    Un = (Bu + (dt * D1) * dLu) ./ (1 + (dt * D1) * lam);
    Vn = (Bv + (dt * D2) * dLv) ./ (1 + (dt * D2) * lam);
  end
end
