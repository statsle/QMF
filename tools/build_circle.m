function data = build_circle(sigma, num)
    theta = linspace(-pi, pi, num);
    data = [cos(theta);sin(theta)];
    data = data + sigma*randn(size(data));
end