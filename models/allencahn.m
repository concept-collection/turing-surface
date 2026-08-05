% Allen-Cahn on a closed surface: interfaces form, then coarsen.
%
%   du/dt = eps2*lap_g(u) + u - u^3
%
% Same scheme as models/schnakenberg.m.

function [U, u] = init(noise)
  U = analys(noise);
  u = synth(U);
end

function [Un, u] = step(U, lam, filt, gx, gy, gz, p1, p2, q2, r, jhat, eps2, dt, niter)
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
    [Ftu, Fpu] = synth(vtu, vpu);
    Pu = p1 .* Ftu + p2 .* Fpu;
    Qu = p2 .* Ftu + q2 .* Fpu;
    PAu = analys(Pu);
    Pcu = PAu .* filt;
    scu = dthetac(Pcu);
    Lu = synth(scu);
    dQu = dphig(Qu);
    lapu = r .* (Lu + dQu);
    dLu = (analys(lapu) + lamJ .* Un) .* filt;

    Un = (Bu + (dt * eps2) * dLu) ./ (1 + (dt * eps2) * lamJ);
  end
end
