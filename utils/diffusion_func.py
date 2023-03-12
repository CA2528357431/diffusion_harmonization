def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    start = scale * 0.0001
    end = scale * 0.02
    return torch.linspace(start, end, timesteps, dtype=torch.float64)