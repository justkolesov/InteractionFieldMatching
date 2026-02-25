import torch
import numpy as np


class Poisson():
    
    
    def __init__(self, config):
        """Construct a PFGM.

        Args:
          config: configurations
        """
        self.config = config
        self.N = config.sampling.N

        
    @property
    def M(self):
        return self.config.training.M

 
    def prior_sampling(self, shape):
        """
        Sampling initial data from p_prior on z=z_max hyperplane.
        See Section 3.3 in PFGM paper
        returns - torch.Size([B,C,H,W])
        """

        # Sample the radius from p_radius (details in Appendix A.4 in the PFGM paper)
        max_z = self.config.sampling.z_max
        N = self.config.data.channels * self.config.data.image_size * self.config.data.image_size + 1
        
        # Sampling form inverse-beta distribution
        samples_norm = np.random.beta(a=N / 2. - 0.5, b=0.5, size=shape[0])
        inverse_beta = samples_norm / (1 - samples_norm)
        
        # Sampling from p_radius(R) by change-of-variable
        samples_norm = np.sqrt(max_z ** 2 * inverse_beta)
        
        # clip the sample norm (radius)
        samples_norm = np.clip(samples_norm, 1, self.config.sampling.upper_norm)
        samples_norm = torch.from_numpy(samples_norm).cuda().view(len(samples_norm), -1)

        # Uniformly sample the angle direction
        gaussian = torch.randn(shape[0], N - 1).cuda()
        unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)

        # Radius times the angle direction
        init_samples = unit_gaussian * samples_norm

        return init_samples.float().view(len(init_samples), self.config.data.num_channels,
                                    self.config.data.image_size, self.config.data.image_size)
   
    
    def ode(self, net_fn, x, t):

        z = np.exp(t.mean().cpu())
        if self.config.sampling.vs:
            print(z)
        x_drift, z_drift = net_fn(x, torch.ones((len(x))).cuda() * z)
        x_drift = x_drift.view(len(x_drift), -1)

        # Substitute the predicted z with the ground-truth
        # Please see Appendix B.2.3 in PFGM paper (https://arxiv.org/abs/2209.11178) for details
        z_exp = self.config.sampling.z_exp
        if z < z_exp and self.config.training.gamma > 0:
            data_dim = self.config.data.image_size * self.config.data.image_size * self.config.data.channels
            sqrt_dim = np.sqrt(data_dim)
            norm_1 = x_drift.norm(p=2, dim=1) / sqrt_dim
            x_norm = self.config.training.gamma * norm_1 / (1 -norm_1)
            x_norm = torch.sqrt(x_norm ** 2 + z ** 2)
            z_drift = -sqrt_dim * torch.ones_like(z_drift) * z / (x_norm + self.config.training.gamma)

        # Predicted normalized Poisson field
        v = torch.cat([x_drift, z_drift[:, None]], dim=1)

        dt_dz = 1 / (v[:, -1] + 1e-5)
        dx_dt = v[:, :-1].view(len(x), self.config.data.num_channels,
                          self.config.data.image_size, self.config.data.image_size)
        dx_dz = dx_dt * dt_dz.view(-1, *([1] * len(x.size()[1:])))
        # dx/dt_prime =  z * dx/dz
        dx_dt_prime = z * dx_dz
        return dx_dt_prime