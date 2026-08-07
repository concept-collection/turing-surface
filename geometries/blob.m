% A random blob: the sphere, radius-modulated by a smooth random function
% on the sphere — surfacefun's blob, built on chebfun's randnfunsphere
% (tools/randnfunsphere.m).
%
% `seed` picks the draw; the same seed always gives the same blob. `scale`
% is the random function's wavelength, so smaller means finer lobes.

function [gx, gy, gz] = shape(theta, phi, amp, scale, seed)
  rng(seed);
  f = randnfunsphere(scale, theta, phi);
  % blob.m's normalization: shift nonnegative, rescale to [-1, 1].
  f = f + abs(min(f));
  f = 2*(f/max(f)) - 1;
  r = 1 + amp*f;
  st = sin(theta);
  gx = r .* (st .* cos(phi));
  gy = r .* (st .* sin(phi));
  gz = r .* cos(theta);
end
