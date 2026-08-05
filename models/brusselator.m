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

function [Un, Vn, u, v] = step(U, V, lam, filt, gx, gy, gz, p1, p2, q2, r, A, B, D1, D2, dt, niter)
  u = synth(U);
  v = synth(V);
  uuv = u .* u .* v;

  Bu = U + dt * analys(A - (B + 1) * u + uuv);
  Bv = V + dt * analys(B * u - uuv);

  Un = Bu ./ (1 + (dt * D1) * lam);
  Vn = Bv ./ (1 + (dt * D2) * lam);

  for k = 1:niter
    % dlap = lap_g - lap_s, evaluated at the current iterate in flux form
    % (see models/schnakenberg.m, docs/richardson-iteration.md and
    % docs/reduced-transforms.md for the derivation).
    Fu = Un .* filt;
    Ftu = synth(dthetac(Fu));
    Fpu = synth(dphic(Fu));
    Pu = p1 .* Ftu + p2 .* Fpu;
    Qu = p2 .* Ftu + q2 .* Fpu;
    Pcu = analys(Pu) .* filt;
    Qcu = analys(Qu) .* filt;
    scu = dthetac(Pcu) + dphic(Qcu);
    lapu = r .* synth(scu);
    dLu = analys(lapu) + lam .* Un;

    Fv = Vn .* filt;
    Ftv = synth(dthetac(Fv));
    Fpv = synth(dphic(Fv));
    Pv = p1 .* Ftv + p2 .* Fpv;
    Qv = p2 .* Ftv + q2 .* Fpv;
    Pcv = analys(Pv) .* filt;
    Qcv = analys(Qv) .* filt;
    scv = dthetac(Pcv) + dphic(Qcv);
    lapv = r .* synth(scv);
    dLv = analys(lapv) + lam .* Vn;

    Un = (Bu + (dt * D1) * dLu) ./ (1 + (dt * D1) * lam);
    Vn = (Bv + (dt * D2) * dLv) ./ (1 + (dt * D2) * lam);
  end
end
