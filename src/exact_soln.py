from math import pi, sqrt

def exact_solution(x , t, D, Q):
  return Q/torch.sqrt(4*np.pi*D*t_plot) * torch.exp(-x_plot**2/(4*D*t_plot))
