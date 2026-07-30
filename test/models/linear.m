% Test model: a purely linear reaction, f(u) = c*u.
%
% Every spherical-harmonic mode then evolves independently under one IMEX Euler
% step, with a closed-form growth factor per degree l:
%
%   U_lm^{n+1} = U_lm^n * (1 + dt*c) / (1 + dt*D*l(l+1))
%
% so a run can be checked against exact arithmetic rather than against another
% implementation. Used by the analytic tests; not offered in the app.

function [U, u] = init(noise)
  U = analys(noise);
  u = synth(U);
end

function [Un, u] = step(U, lam, c, D, dt)
  u = synth(U);
  Un = (U + dt * analys(c * u)) ./ (1 + (dt * D) * lam);
end
