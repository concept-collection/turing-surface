% Brusselator reaction-diffusion on a closed surface.
%
%   du/dt = D1*lap_g(u) + A - (B+1)*u + u^2*v
%   dv/dt = D2*lap_g(v) +     B*u     - u^2*v
%
% Same scheme as models/schnakenberg.m, including the grouped transforms:
% [a, b] = synth(x, y) runs the group as batched Legendre dispatches.

% Seeded from a smooth random field -- see models/schnakenberg.m.
function [U, V, u, v] = init(lam3, gx, gy, gz, A, B)
  f = randnfun3(lam3, gx, gy, gz);
  [U, V] = analys(A + 0.01*f, (B / A) * ones(numel(f), 1));
  [u, v] = synth(U, V);
end

function [Un, Vn, u, v] = step(U, V, lam, filt, gx, gy, gz, p1, p2, q2, r, jhat, A, B, D1, D2, dt, niter)
  [u, v] = synth(U, V);
  uuv = u .* u .* v;

  ru = A - (B + 1) * u + uuv;
  rv = B * u - uuv;
  [Ru, Rv] = analys(ru, rv);
  Bu = U + dt * Ru;
  Bv = V + dt * Rv;

  % Mean-J preconditioning -- see models/schnakenberg.m.
  lamJ = lam ./ jhat;
  Un = Bu ./ (1 + (dt * D1) * lamJ);
  Vn = Bv ./ (1 + (dt * D2) * lamJ);

  for k = 1:niter
    % dlap = lap_g - lap_s, evaluated at the current iterate in flux form
    % (see models/schnakenberg.m, docs/richardson-iteration.md and
    % docs/reduced-transforms.md for the derivation and the ordering).
    Fu = Un .* filt;
    Fv = Vn .* filt;
    vtu = dthetac(Fu);
    vpu = dphic(Fu);
    vtv = dthetac(Fv);
    vpv = dphic(Fv);
    [Ftu, Fpu, Ftv, Fpv] = synth(vtu, vpu, vtv, vpv);
    Pu = p1 .* Ftu + p2 .* Fpu;
    Qu = p2 .* Ftu + q2 .* Fpu;
    Pv = p1 .* Ftv + p2 .* Fpv;
    Qv = p2 .* Ftv + q2 .* Fpv;
    [PAu, PAv] = analys(Pu, Pv);
    Pcu = PAu .* filt;
    Pcv = PAv .* filt;
    scu = dthetac(Pcu);
    scv = dthetac(Pcv);
    [Lu, Lv] = synth(scu, scv);
    dQu = dphig(Qu);
    dQv = dphig(Qv);
    lapu = r .* (Lu + dQu);
    lapv = r .* (Lv + dQv);
    [LAu, LAv] = analys(lapu, lapv);
    dLu = (LAu + lamJ .* Un) .* filt;
    dLv = (LAv + lamJ .* Vn) .* filt;

    Un = (Bu + (dt * D1) * dLu) ./ (1 + (dt * D1) * lamJ);
    Vn = (Bv + (dt * D2) * dLv) ./ (1 + (dt * D2) * lamJ);
  end
end
