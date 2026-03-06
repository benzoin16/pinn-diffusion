def loss_pde(model, x, t, D):
    x.requires_grad_(True)
    t.requires_grad_(True)
    C = model(x, t)
    C_t = torch.autograd.grad(C, t, grad_outputs=torch.ones_like(C), create_graph=True)[0]
    C_x = torch.autograd.grad(C, x, grad_outputs=torch.ones_like(C),create_graph=True)[0]
    C_xx = torch.autograd.grad(C_x, x, grad_outputs=torch.ones_like(C_x),create_graph=True)[0]
    residual = C_t - D * C_xx
    return torch.mean(residual**2)

def loss_ic(model, x, Q, sigma):
    t = torch.zeros_like(x)
    C_pred = model(x, t)
    C_true = Q/(np.sqrt(2*np.pi)*sigma) * torch.exp(-(x**2)/(2*sigma**2))
    return torch.mean((C_pred - C_true)**2)

def loss_bc(model, t, L):
    x_left  = -L * torch.ones_like(t)
    x_right =  L * torch.ones_like(t)
    C_left  = model(x_left, t)
    C_right = model(x_right, t)
    return torch.mean(C_left**2 + C_right**2)

def loss_mass(model, x, t, Q):
    C = model(x, t)
    integral = torch.trapz(C.squeeze(), x.squeeze())
    return (integral - Q)**2

