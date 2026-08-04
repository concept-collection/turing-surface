% Schnakenberg reaction-diffusion on a closed surface.
%
%   du/dt = D1*lap_g(u) + a - u + u^2*v
%   dv/dt = D2*lap_g(v) + b     - u^2*v
%
% Explicit reaction, implicit diffusion (IMEX Euler). The implicit solve
% splits lap_g = lap_s + dlap: the round-sphere part lap_s is diagonal in
% spherical-harmonic space (eigenvalues -lam), and the loop iterates the
% geometric correction dlap from that exact solve. Grid fields are npts x 1;
% spectral fields are real 2 x nlm. See docs/richardson-iteration.md.

function [U, V, u, v] = init(noise, a, b)
  us = a + b;
  vs = b / (us * us);
  U = analys(us + noise);
  V = analys(vs * ones(numel(noise), 1));
  u = synth(U);
  v = synth(V);
end

function [Un, Vn, u, v] = step(U, V, lam, filt, gx, gy, gz, Vtx, Vty, Vtz, Vpx, Vpy, Vpz, a, b, D1, D2, dt, niter)
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
    % dlap = lap_g - lap_s, evaluated at the current iterate (Algorithm 3 of
    % evolving_surface/notes/algos.tex): surface gradient of the field,
    % contracted through the inverse metric quantities Vt*/Vp*; each
    % Cartesian component re-analysed and differentiated again; recombined
    % into the surface divergence. lam.*Un adds back -lap_s(Un), since lam
    % holds +l(l+1). filt zeroes the top two degrees, where the theta/phi
    % derivative recurrences cannot exactly represent a derivative. See
    % docs/richardson-iteration.md.
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
