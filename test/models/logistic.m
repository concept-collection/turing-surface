% Test model: a nonlinear reaction, f(u) = r*u*(1 - u).
%
% Started from a *uniform* field, the state stays uniform, and diffusion does
% nothing to it (the l = 0 eigenvalue is zero). So every step is exactly the
% explicit Euler map of the scalar ODE,
%
%   u^{n+1} = u^n + dt*r*u^n*(1 - u^n)
%
% which checks that the generated kernel evaluates a nonlinear reaction
% correctly, against arithmetic rather than another implementation. Used by the
% analytic tests; not offered in the app.
%
% The caller passes the initial grid field as `noise` (the name the app uses for
% its seeded perturbation); here the tests put an exact field there.

function [U, u] = init(noise)
  U = analys(noise);
  u = synth(U);
end

function [Un, u] = step(U, lam, r, D, dt)
  u = synth(U);
  Un = (U + dt * analys(r * u .* (1 - u))) ./ (1 + (dt * D) * lam);
end
