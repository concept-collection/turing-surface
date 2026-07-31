% A dumbbell: radius 1 - waist*sin(theta)^2, stretched along z.

function [gx, gy, gz] = shape(theta, phi, waist, stretch)
  st = sin(theta);
  r = 1 - waist * (st .^ 2);
  gx = r .* (st .* cos(phi));
  gy = r .* (st .* sin(phi));
  gz = (1 + stretch) * (r .* cos(theta));
end
