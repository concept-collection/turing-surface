% Allen-Cahn on a closed surface: interfaces form, then coarsen.
%
%   du/dt = eps2*lap_g(u) + u - u^3
%
% Same scheme as models/schnakenberg.m, sphere-split flux divergence included.

% Seeded from a smooth random field -- see models/schnakenberg.m.
function [U, u] = init(lam3, gx, gy, gz)
  U = analys(0.01 * randnfun3(lam3, gx, gy, gz));
  u = synth(U);
end

function [Un, u] = step(U, lam, filt, gx, gy, gz, p2, r, dp1, dq2, jinv, jhat, eps2, dt, niter)
  u = synth(U);

  Bu = U + dt * analys(u - u.^3);

  % Mean-J preconditioning -- see models/schnakenberg.m.
  lamJ = lam ./ jhat;
  Un = Bu ./ (1 + (dt * eps2) * lamJ);

  for k = 1:niter
    % dlap = lap_g - lap_s, evaluated at the current iterate in flux form
    % (see models/schnakenberg.m, docs/richardson-iteration.md and
    % docs/reduced-transforms.md for the derivation; the grouped calls run
    % the gradient syntheses and the flux analyses as batched dispatches).
    Fu = Un .* filt;
    vtu = dthetac(Fu);
    vpu = dphic(Fu);
    [Ftu, Fpu, Su] = synth(vtu, vpu, lam .* Fu);
    Pu = dp1 .* Ftu + p2 .* Fpu;
    Qu = p2 .* Ftu + dq2 .* Fpu;
    PAu = analys(Pu);
    Pcu = PAu .* filt;
    scu = dthetac(Pcu);
    Lu = synth(scu);
    dQu = dphig(Qu);
    lapu = r .* (Lu + dQu) - jinv .* Su;
    dLu = (analys(lapu) + lamJ .* Un) .* filt;

    Un = (Bu + (dt * eps2) * dLu) ./ (1 + (dt * eps2) * lamJ);
  end
end
