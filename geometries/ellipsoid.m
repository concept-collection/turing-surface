% A triaxial ellipsoid: the sphere with each axis scaled independently.

function [gx, gy, gz] = shape(theta, phi, ax, ay, az)
  st = sin(theta);
  gx = ax * (st .* cos(phi));
  gy = ay * (st .* sin(phi));
  gz = az * cos(theta);
end
