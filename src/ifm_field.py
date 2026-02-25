import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import clear_output

import ot
import math
import typing as tp
from tqdm import tqdm

#import wandb
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append("/trinity/home/a.kolesov/EFM/")
from src.ode import get_rk45_sampler_pfgm, LearnedImageODESolver


class IFM:
    
    def __init__(self, config):
        self._config = config
        
    @property
    def config(self):
        return self._config
    
    @config.setter
    def config(self, config):
        print("You modify the configuration for code running")
        self._config = config
        
    def __str__(self):
        print("IFM")
    
    def __repr__(self):
        return f"IFM({self._config})"
    
    
    ######################################
    @staticmethod
    def plotVectorField(*,mesh: torch.Tensor,
                         field:torch.Tensor, 
                         p_samples:torch.Tensor,
                         q_samples:torch.Tensor, **kwargs: tp.Any) -> matplotlib.figure.Figure:
        
        
        
        fig = plt.figure()
        fig.set_figheight(kwargs.get('figheight', 7))
        fig.set_figwidth( kwargs.get('figwidth', 7))

        ax = fig.add_subplot(1,1,1, projection='3d' )
        ax.scatter(p_samples[:,0].cpu(),p_samples[:,1].cpu(),p_samples[:,2].cpu(),
                   color='blue',edgecolor='black',s=80,label=r'$x_{+}  \sim \mathbb{P}(x_{+})$')
        ax.scatter(q_samples[:,0].cpu(),q_samples[:,1].cpu(),q_samples[:,2].cpu(),
                   color='red',edgecolor='black',s=80,label=r'$x_{-}  \sim \mathbb{Q}(x_{-})$')
        ax.quiver(mesh[:, 0].cpu(), mesh[:, 1].cpu(), mesh[:,2].cpu(),
                  field[:, 0].cpu(), field[:, 1].cpu(), field[:,2].cpu(),
                  color='black',length=1, normalize=True)
        ax.set_title(kwargs.get("title", "EFM Ground Truth"))
        ax.legend()
        return fig
    ######################################## 
    
    
    
    
    ######################################## 
    @staticmethod
    def plotTrajectories(*,traj: torch.Tensor,  
                         p_samples: torch.Tensor,
                         q_samples: torch.Tensor,**kwargs: tp.Any) -> matplotlib.figure.Figure:
        
        fig = plt.figure()
        fig.set_figheight(kwargs.get('figheight', 7))
        fig.set_figwidth( kwargs.get('figwidth', 7))
        
        ax = fig.add_subplot(1,1,1,  projection='3d' )
        ax.scatter(p_samples[:,0].cpu(),p_samples[:,1].cpu(),p_samples[:,2].cpu(),
                   color='blue',edgecolor='black',s=80, label=r'$x_{+}  \sim \mathbb{P}(x_{+})$')
        
         
        
        traj[-1][:,0] = 6.5
        for jdx in range(len(traj[-1])):
            
            ax.scatter(traj[-1][jdx,0].cpu(),traj[-1][jdx,1].cpu(),traj[-1][jdx,2].cpu(),
            color='lightgreen',edgecolor='black',zorder=20,label=r'$y \sim  T(x_{+})$' if jdx==0 else None,s=80)
            
            ax.scatter(q_samples[jdx,0].cpu(),q_samples[jdx,1].cpu(),q_samples[jdx,2].cpu(),
                   color='red',edgecolor='black',s=80, label=r'$x_{-}  \sim \mathbb{Q}(x_{-})$' if jdx==0 else None)
            
        for idx in range(200,230):
            ax.plot(traj[: ,idx,0].cpu(), 
                traj[:, idx,1].cpu(),
                traj[:, idx,2].cpu(),
                color='black',linewidth=0.5, zorder=3);
        ax.set_title(kwargs.get("title", "EFM Ground Truth"))
        ax.legend() 
        return fig
    ######################################## 
    
    
    
    ######################################
    @staticmethod
    def plot(x: torch.Tensor, **kwargs: tp.Any ) -> matplotlib.figure.Figure:
        
        fig,ax = plt.subplots(kwargs.get('figsize',5),
                              kwargs.get('figsize',5),
                              figsize=(kwargs.get('figsize',5),
                                       kwargs.get('figsize',5)))
        
        for idx in range(kwargs.get('figsize',5)):
            for jdx in range(kwargs.get('figsize',5)):
                ax[idx,jdx].imshow(x[idx,jdx])
                ax[idx,jdx].set_yticks([])
                ax[idx,jdx].set_xticks([])
        fig.tight_layout(pad=kwargs.get('pad',0.00001))       
        return fig
    ######################################

    

    ######################################
    @staticmethod
    def plota(x: torch.Tensor, **kwargs: tp.Any ) -> matplotlib.figure.Figure:
        
        fig,ax = plt.subplots(kwargs.get('figsize',5),
                              kwargs.get('figsize',5),
                              figsize=(kwargs.get('figsize',5),
                                       kwargs.get('figsize',5)))
        
        for idx in range(kwargs.get('figsize',5)):
            for jdx in range(kwargs.get('figsize',5)):
                ax[idx,jdx].imshow(x[idx,jdx].permute(1,2,0).cpu())
                ax[idx,jdx].set_yticks([])
                ax[idx,jdx].set_xticks([])
        fig.tight_layout(pad=kwargs.get('pad',0.00001))       
        return fig
    ######################################
    
    
    
    ######################################
    @staticmethod
    def plot_trajectory(traj: torch.Tensor, **kwargs: tp.Any) -> matplotlib.figure.Figure:
    
        fig,ax = plt.subplots(kwargs.get('figsize',5),
                              len(traj),
                              figsize=(len(traj),kwargs.get('figsize',5)),
                              sharex=True,sharey=True)
        
        for time in range(len(traj)):
            for idx in range(kwargs.get('figsize',5)):
                ax[idx,time].imshow(np.clip(traj[time,idx].permute(1,2,0).cpu().numpy()*255,0,255).astype(np.uint8))
                ax[idx,time].set_xticks([])
                ax[idx,time].set_yticks([])

        fig.tight_layout(pad=kwargs.get('pad',0.00001))
        return fig
    ######################################
    
    
    
    
    ########################################
    @staticmethod
    def get_mesh( kv : tp.Dict[str, float], 
                  mesh_num_points: tp.Optional[int] = None) -> torch.Tensor:
  
        num_points = 10 if mesh_num_points is None else mesh_num_points
    
        if len(kv) == 3:
            mesh = []
            z = torch.linspace(kv["z"][0],kv["z"][1], mesh_num_points)
            x = torch.linspace(kv["x"][0], kv["x"][1], mesh_num_points)
            y = torch.linspace(kv["y"][0], kv["y"][1], mesh_num_points)

            for i in x:
                for j in y:
                    for k in z:
                        mesh.append(torch.tensor([i,j,k]))
        else:
            raise ValueError("The length of DICT MESH should be 3!")
        return torch.stack(mesh, dim=0)
    ########################################
    
    
    
    
    #######  get scale ########
    def get_scale(self, z_proj, l_prime ):

        """
        z_proj - torch.array([B])
        l_prime - torch.array([1])

        returns: torch.array([B])
        """
        #print(z_proj)
        #assert np.logical_and(z_proj >= 0,z_proj <= l_prime).all() == True
        scale = -100.*torch.ones((z_proj.shape[0],1)).to(self._config.device) #[B]
        mask = torch.logical_and(z_proj < self._config.D, z_proj >= 0)  #[B]
        scale[mask] = self._config.SCALE*torch.sin(self._config.K*z_proj[mask])
        mask = torch.logical_and(z_proj > l_prime - self._config.D, z_proj <= l_prime ) #[B]
        scale[mask] = self._config.SCALE*torch.cos(self._config.K*(z_proj[mask] - l_prime + self._config.D))
        mask = torch.logical_and(z_proj >= self._config.D,  z_proj <= l_prime - self._config.D)  #[B]  # straight part 
        scale[mask] = self._config.SCALE
        mask =  torch.logical_or(z_proj < 0, z_proj > l_prime)
        scale[mask] = 1e-18

        assert torch.any(scale == -100.) == False

        return scale  
    #############################    



     
    ###### get direction #########
    def get_direction(self, x_proj, z_proj, normal_quark, 
                      n_orthog, l_prime ):

        """
        x_proj - torch.array([B,1])
        z_proj - torch.array([B,1])
        normal_quark - torch.array([1,DIM])
        n_orthog - torch.array([B, DIM])
        l_prime - value

        returns: torch.array([B,DIM]), torch.array([B,1])
        """

        #assert np.logical_and(z_proj >= 0,z_proj <= l_prime).all() == True
        direction =  -100*torch.ones([z_proj.shape[0],self._config.DIM]).to(self._config.device) #[B, DIM]
        alpha = -100*torch.ones((z_proj.shape[0],1)).to(self._config.device) #[B,1]

        mask = torch.logical_and(z_proj < self._config.D, z_proj >= 0).reshape(-1)  #[B]
        alpha[mask] = torch.arctan(x_proj[mask]*self._config.K/(torch.tan(self._config.K*z_proj[mask])  + 1e-5)) 
        direction[mask] = torch.cos(alpha[mask].reshape(-1,1))*normal_quark + torch.sin(alpha[mask].reshape(-1,1))*n_orthog[mask] 

        mask =  torch.logical_and(z_proj > l_prime - self._config.D, z_proj <= l_prime).reshape(-1)
        alpha[mask] = torch.arctan(x_proj[mask]*self._config.K*torch.tan(math.pi - self._config.K*(z_proj[mask] - l_prime + self._config.D)))
        direction[mask] = torch.cos(alpha[mask].reshape(-1,1))*normal_quark + torch.sin(alpha[mask].reshape(-1,1))*n_orthog[mask]

        mask = torch.logical_and(z_proj >= self._config.D, z_proj <= l_prime - self._config.D) .reshape(-1)
        direction[mask] = normal_quark
        alpha[mask] = torch.zeros((1,1)).to(self._config.device)

        mask = torch.logical_or(z_proj < 0, z_proj > l_prime).reshape(-1) 

        direction[mask] = torch.zeros((1,self._config.DIM)).to(self._config.device) 
        alpha[mask] = torch.zeros((1,1)).to(self._config.device)

        assert torch.any(direction[:,0] == -100.) == False

        return direction, alpha
    ############################# 
    
    
    

    ###### get field ######
    def get_field(self, q, a_q, points ):

        """
        q - torch.array([1, DIM])
        a_q - torch.array([1,DIM])
        points - torch.array([B, DIM])

        returns - torch.array([B, DIM])
        """

        distance = a_q - q #[1,DIM]
        norm_dist = torch.norm(distance, dim=1).reshape(-1,1) #[1,1]
        L_prime = norm_dist.item()
        normal_quark = distance/ norm_dist #[1,DIM]

 
        if self._config.training.field_type == "Unshifted":

            r_prime = points - q #[B,DIM]
            z_proj = torch.mul(r_prime, normal_quark ).sum(dim=1).reshape(-1,1) #[B,1]
            r_orthog = r_prime - z_proj*normal_quark #[B,DIM]
            n_orthog = r_orthog/ ( (torch.norm(r_orthog,dim=1) + 1e-5).reshape(-1,1) ) #[B,DIM]
            x_proj =  torch.diag( torch.matmul(r_prime, n_orthog.T) ).reshape(-1,1) #[B,1]
            scale = self.get_scale(z_proj, L_prime) #[B,1]
            normal_E, alpha =  self.get_direction(x_proj, z_proj, normal_quark, n_orthog, L_prime)
            field = normal_E*(torch.exp(-x_proj**2/(2*scale**2))/ (torch.cos(alpha)*scale**(self._config.DIM-1)))
            
            
        if self._config.training.field_type == "Shifted":
            z_proj = points[:,0][:, None] #[1,DIM]
            rho = (points - q) - distance*(z_proj/self._config.L) #[1,DIM]
            r_orthog = torch.norm(rho, dim=1)[:,None]
            n_orthog = rho/(r_orthog + 1e-11)
            scale = self.get_scale(z_proj, self._config.L ) #[B,1]
            normal_E, alpha =  self.get_direction( r_orthog, z_proj, normal_quark, n_orthog, self._config.L)
            field  = normal_E*(torch.exp(-r_orthog**2/(2*scale**2)))/((torch.cos(alpha) + 1e-8)*(scale/self._config.SCALE)**(self._config.DIM-1) + 1e-2)
 
        else:
            raise ValueError(f"There is no such field type as {self._config.field_type}")

        return field
    ##########################



   

    ###########################
    def GroundTruth(self, points, quarks, anti_quarks):

        """
        quarks - np.array([B_1,DIM])
        anti_quarks - np.array([B_2,DIM])
        points - np.array([B,DIM])

        returns: np.array([B,DIM])
        """
   
        ### exceptional experiment ###
        if self._config.data.name == "2d_discrete":
            field = torch.zeros((points.shape[0], self._config.DIM)).to(self._config.device) #[B, DIM]
            # torch.cdist for the future implementation
            for q in quarks:
                for a_q in anti_quarks:
                    field += self.get_field(q.reshape(1,self._config.DIM), a_q.reshape(1,self._config.DIM) ,
                                       points)
            return field
        ### exceptional experiment ###



        if self._config.training.plan_type == "Optimal":
            assert quarks.shape[0] == anti_quarks.shape[0]
            ot_emd = ot.da.EMDTransport(metric = 'euclidean')
            dot = ot_emd.fit(Xs =  quarks.view(-1, self._config.DIM).cpu().numpy(),
                             Xt = anti_quarks.view(-1, self._config.DIM).cpu().numpy())
            map_anti_quarks = torch.from_numpy(ot_emd.transform(Xs=quarks.view(-1, 
                              self._config.DIM).cpu().numpy())).to(self._config.device)

        elif self._config.training.plan_type == "Independent":
            map_anti_quarks = anti_quarks
            
        else:
            raise ValueError(f"There is no such plan type as {self._config.plan_type}")

            
        field = torch.zeros((points.shape[0], self._config.DIM)).to(self._config.device)
        for (q, a_q) in zip(quarks, map_anti_quarks):
            field += self.get_field(q.reshape(1,self._config.DIM), a_q.reshape(1,self._config.DIM) ,
                                   points )
            
        return  field 
    ###########################
    
    
    
    ######################################## 
    def forward_interpolation(self,
                              p_samples: torch.Tensor,
                              q_samples: torch.Tensor) -> torch.Tensor:
        
        """
        The definition Inter-plate points between plates.
        
        Input:
        p_samples - torch.Size([b,D+1]) or torch.Size([b,C,H,W])
        q_samples - torch.size([b,D+1]) or torch.Size([b,C,H,W])
        
        Return:
        perturbed_vec_samples - torch.Size([B,D+1])
        """
        
        

        #################################################
        ######       Mesh Interpolation (Toy)      ######
        #################################################
        if self._config.training.interpolation == 'mesh':
            
            mesh = self.get_mesh(self._config.KV, self._config.mesh_num_points).to(self._config.device)
            idxs = torch.randperm(mesh.shape[0])
            perturbed_samples_vec = mesh[idxs][:self._config.training.small_batch_size]
        #################################################
        ######   Mesh Interpolation (Toy)          ######
        #################################################   
        

        
        
        #################################################
        ######   Uniform Interpolation             ######
        ################################################# 
        elif self._config.training.interpolation == 'Uniform':

            t = torch.distributions.Uniform(low=self._config.p.x_loc + self._config.training.epsilon,
                high=self._config.q.x_loc - self._config.training.epsilon).sample(\
                torch.Size([self._config.training.small_batch_size])).to(self._config.device)[:, None] #[b,1]
            
            """
            den = self._config.L - 2*self._config.training.epsilon
            perturbed_x = (t-self._config.training.epsilon)/den*\
                          q_samples[:self._config.training.small_batch_size,1:]\
               +(1-(t-self._config.training.epsilon)/den)*\
               p_samples[:self._config.training.small_batch_size,1:] #[b, D+1]
            """
            perturbed_x = q_samples*(t[:,None,None]/self._config.L) + (1 - t[:,None,None]/self._config.L)*p_samples
           
             
            if self._config.training.noised_interpolation:
                
                perturbed_x = perturbed_x.reshape(len(p_samples), self._config.DIM-1)
                ###################
                gaussian = torch.randn(p_samples.shape[0], self._config.DIM-1).to(p_samples.device) # torch.Size([b, C*H*W=D])
                unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True) #  torch.Size([b, C*H*W=D])
                
                #mltp = 2*torch.log(1 - torch.abs(t.squeeze()-self._config.L/2) + self._config.L/2)
                mltp = 1 + torch.cos(2*math.pi/self._config.L*(t.squeeze()-self._config.L/2))
                
                perturbed_x = perturbed_x + self._config.training.noised_interpolation_mltp*unit_gaussian*mltp[:,None] 
                
                
            perturbed_samples_vec = torch.cat([t, perturbed_x.view(self._config.training.small_batch_size,
                                                                   self._config.DIM-1)], dim=1) 
        #################################################
        ######   Uniform Interpolation             ######
        ################################################# 



        
        #################################################
        ######           PFGM Interpolation        ######
        #################################################
        elif self._config.training.interpolation == 'Gaussian_mixing':

            assert p_samples.shape == torch.Size([self._config.training.small_batch_size, self._config.data.num_channels,
                                                  self._config.data.image_size, self._config.data.image_size]) 
            assert p_samples.shape == q_samples.shape
            
            ####### perturbation for z component #######
            m = torch.randn((p_samples.shape[0],), device=p_samples.device) * self._config.training.M  #[b] : m ~ U[0,M]
            tau = self._config.training.tau 
            z = torch.randn( p_samples.shape[0], 1).to(p_samples.device)*self._config.training.sigma_end #[b, 1] : influence on the performance ??
            z = z.abs()  # [b,1]  : z = epsilon + N(0,I)
            assert z.shape==torch.Size([self._config.training.small_batch_size, 1])
                                      
            # confine norms
            # see Appendix B.1.1 of https://arxiv.org/abs/2209.11178
            """
            if config.training.restrict_M:
                idx = (z < self._config.training.epsilon + 0.005).squeeze()
                num = int(idx.int().sum())
                restrict_m = int(self._config.training.M * 0.7)
                m[idx] = torch.rand((num,), device=p_samples.device) * restrict_m
            """
            if self._config.training.restrict_M:
                idx = (z < 0.005).squeeze()
                num = int(idx.int().sum())
                restrict_m = int(self._config.training.M * 0.7)
                m[idx] = torch.rand((num,), device=p_samples.device) * restrict_m
            
 
            multiplier = (1+tau) ** m # torch.Size([b]) : the essence of this form??
            perturbed_z = z.squeeze() * multiplier   # torch.Size([b])* torch.Size([b]) = 
            
            
            perturbed_z = torch.clamp(perturbed_z, min=self._config.training.epsilon ,
                                                   max=self._config.L - self._config.training.epsilon) #torch.Size([b])
            
            
            ####### perturbation for z component #######
            
            
            ####### perturbation for x component #######
            # Sample uniform angle
            gaussian = torch.randn(p_samples.shape[0], self._config.DIM-1).to(p_samples.device) # torch.Size([b, C*H*W=D])
            unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True) #  torch.Size([b, C*H*W=D])
            noise = torch.randn_like(p_samples).reshape(p_samples.shape[0] , 
                                                        -1) * self._config.training.sigma_end #torch.Size([b, C*H*W=D])
            norm_m = torch.norm(noise, p=2, dim=1) * multiplier  # torch.Size([b])*torch.Size([b]) = torch.Size([b])

            # Construct the perturbation for x
            perturbation_x = unit_gaussian*norm_m[:, None] # torch.Size([b,C*H*W])* torch.Size([b,1])=  torch.Size([b,C*H*W=D])
            perturbation_x = perturbation_x.view_as(p_samples) # torch.size([b,C,H,W])
            
            """
            perturbed_x =  q_samples*(perturbed_z[:,None,None,None]/self._config.L) + (1 - perturbed_z[:,None,None,None]/self._config.L)*p_samples  #torch.size([b,C,H,W])  + torch.size([b,C,H,W])
            """
            perturbed_x = p_samples + perturbation_x
            
             
             
            ####### perturbation for x component #######
            
            perturbed_samples_vec = torch.cat((perturbed_z[:, None],
                                               perturbed_x.reshape(p_samples.shape[0], self._config.DIM - 1),
                                               ), dim=1) #[b, D+1]
            
        #################################################
        ######           PFGM Interpolation        ######
        #################################################

        
        
        elif self._config.training.interpolation == 'Uniform_mixing':
            
            
            assert p_samples.shape == torch.Size([self._config.training.small_batch_size, self._config.data.num_channels,
                                                  self._config.data.image_size, self._config.data.image_size]) 
            assert p_samples.shape == q_samples.shape
            
            ####### perturbation for z component #######
            m = torch.rand((p_samples.shape[0],), device=p_samples.device) * self._config.training.M  #[b] : m ~ U[0,M]
            tau = self._config.training.tau 
            z = torch.randn( p_samples.shape[0], 1).to(p_samples.device)*self._config.training.sigma_end #[b, 1] : influence on the performance ??
            z = z.abs()  # [b,1]  : z = epsilon + N(0,I)
            assert z.shape==torch.Size([self._config.training.small_batch_size, 1])
                                      
            # confine norms
            # see Appendix B.1.1 of https://arxiv.org/abs/2209.11178
            """
            if config.training.restrict_M:
                idx = (z < self._config.training.epsilon + 0.005).squeeze()
                num = int(idx.int().sum())
                restrict_m = int(self._config.training.M * 0.7)
                m[idx] = torch.rand((num,), device=p_samples.device) * restrict_m
            """
            if self._config.training.restrict_M:
                idx = (z < 0.005).squeeze()
                num = int(idx.int().sum())
                restrict_m = int(self._config.training.M * 0.7)
                m[idx] = torch.rand((num,), device=p_samples.device) * restrict_m
            
 
            multiplier = (1+tau) ** m # torch.Size([b]) : the essence of this form??
            perturbed_z = z.squeeze() * multiplier # torch.Size([b])* torch.Size([b]) = torch.Size([b])
            
            perturbed_z = torch.clamp(perturbed_z, min=self._config.training.epsilon ,
                                                   max=self._config.L - self._config.training.epsilon) #torch.Size([b])
            
            
            perturbed_x = q_samples*(perturbed_z[:,None,None,None]/self._config.L) + (1 - perturbed_z[:,None,None,None]/self._config.L)*p_samples
            
            
            if self._config.training.noised_interpolation:

                perturbed_x = perturbed_x.reshape(len(p_samples), self._config.DIM-1)
                ###################
                gaussian = torch.randn(p_samples.shape[0], self._config.DIM-1).to(p_samples.device) # torch.Size([b, C*H*W=D])
                unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True) #  torch.Size([b, C*H*W=D])
                
                #mltp = 2*torch.log(1 - torch.abs(t.squeeze()-self._config.L/2) + self._config.L/2)
                mltp = 1 + torch.cos(2*math.pi/self._config.L*(t.squeeze()-self._config.L/2))
                
                perturbed_x = perturbed_x + self._config.training.noised_interpolation_mltp*unit_gaussian*mltp[:,None]
                ##################
            
            
            
            
            perturbed_samples_vec = torch.cat([perturbed_z[:, None],
                                               perturbed_x.reshape(len(p_samples), self._config.DIM-1)], dim=1)
            
         
        elif self._config.training.interpolation == 'both_side':
            
            m = torch.rand((p_samples.shape[0],), device=p_samples.device) * self._config.training.M
            z = torch.randn((len(p_samples), 1, 1, 1)).to(p_samples.device) * self._config.training.sigma_end
            z = z.abs()
            data_dim = self._config.data.num_channels * self._config.data.image_size * self._config.data.image_size
            multiplier = (1+self._config.training.tau) ** m
            perturbed_z = z.squeeze() * multiplier + self._config.training.epsilon

            mask_right = torch.nonzero(perturbed_z > self._config.q.x_loc - self._config.training.epsilon)
            perturbed_z[mask_right.view(-1)] = torch.distributions.Uniform(low=self._config.p.x_loc + self._config.training.epsilon,
                    high=self._config.q.x_loc - self._config.training.epsilon).sample(torch.Size([len(mask_right)])).to(self._config.device)

            mask_left = torch.nonzero(perturbed_z < self._config.p.x_loc + self._config.training.epsilon)
            perturbed_z[mask_left.view(-1)] = torch.distributions.Uniform(low=self._config.p.x_loc + self._config.training.epsilon,
                    high=self._config.q.x_loc - self._config.training.epsilon).sample(torch.Size([len(mask_left)])).to(self._config.device)


            
            perturbed_x = q_samples*(perturbed_z[:,None,None,None]/self._config.L) + (1 - perturbed_z[:,None,None,None]/self._config.L)*p_samples
            perturbed_samples_vec = torch.cat([perturbed_z[:, None],
                                               perturbed_x.reshape(len(p_samples), self._config.DIM-1)], dim=1)

        return perturbed_samples_vec
    ######################################## 
    
    
    
    #######################################
    def train(self, train_loader, eval_loader, net, optimizer, optimize_fn,
                    state,  **kwargs: tp.Any):
        
         

        if self._config.experiment == 'generation':
            train_iter = iter(train_loader)
            eval_iter = iter(eval_loader)
            
        elif self._config.experiment == 'translation':
            
            train_iter,eval_iter = {},{}
            for name_set in self._config.data.name_sets:
                train_iter[name_set] = iter(train_loader[name_set])
                eval_iter[name_set]  = iter(train_loader[name_set])
        else:
            raise ValueError
        
        
        
        for step in tqdm(range(self._config.training.n_iters  + 1)):
            
            if self._config.experiment == 'generation':
                try:
                    batch_x,_ =  next(train_iter)
                except StopIteration:
                    print('stop')
                else:
                    train_iter = iter(train_loader)
                    if self._config.model.class_cond:
                        batch_x,cond_x =  next(train_iter)
                        cond_x = cond_x.to(self._config.device)
                    else:
                        batch_x,_ =  next(train_iter) 
                batch_x = batch_x.to(self._config.device)
                batch_y = torch.randn_like(batch_x).to(self._config.device)
                
            elif self._config.experiment == 'translation':
                
                for idx, name_set in enumerate(self._config.data.name_sets):
                    try:
                        batch,_ =  next(train_iter[name_set])
                    except StopIteration:
                        print('stop')
                    else:
                        train_iter[name_set] = iter(train_loader[name_set])
                        batch,_ =  next(train_iter[name_set])
                    batch = batch.to(self._config.device)
                    
                    if idx == 0:
                        batch_x = batch
                    else:
                        batch_y = batch
                        
            else:
                raise ValueError
            
            
            
            if self._config.training.plan_type == "Optimal":
                
                ot_emd = ot.da.EMDTransport(metric = 'euclidean')
                dot = ot_emd.fit(Xs = batch_x.view(-1, self._config.DIM-1).cpu().numpy(),
                                 Xt = batch_y.view(-1, self._config.DIM-1).cpu().numpy() )
                batch_y = torch.from_numpy(ot_emd.transform(Xs=batch_x.view(-1, 
                          self._config.DIM-1).cpu().numpy())).to(self._config.device)
                
            elif self._config.training.plan_type == "Independent":
                pass
            else:
                raise ValueError
            
            batch_y = batch_y.view(-1, self._config.data.num_channels,
                                       self._config.data.image_size,self._config.data.image_size)
            batch_x = batch_x.view(-1, self._config.data.num_channels,
                                       self._config.data.image_size,self._config.data.image_size)
           
            optimizer = state['optimizer']
            optimizer.zero_grad()
            
            
            perturbed_samples_vec = self.forward_interpolation(batch_x[:self._config.training.small_batch_size],
                                                               batch_y[:self._config.training.small_batch_size])
            
            
            field = self.GroundTruth(perturbed_samples_vec,
                                     torch.cat([self._config.p.x_loc*torch.ones(len(batch_x))[:,None].to(self._config.device),
                                                  batch_x.view(-1,self._config.DIM-1)], dim=1),
                                     torch.cat([self._config.q.x_loc*torch.ones(len(batch_y))[:,None].to(self._config.device),
                                                  batch_y.view(-1,self._config.DIM-1)], dim=1))
            
            field = math.sqrt(self._config.DIM)*field/( torch.norm(field, dim=1, keepdim=True) + 1e-5)
            
            
            if kwargs.get(("wandb",False)):
                wandb.log({"field": field.mean().item()}, step=step)
             
            
            perturbed_samples_x = perturbed_samples_vec[:, 1:].view(-1,self._config.data.num_channels,
                                                                    self._config.data.image_size,
                                                                    self._config.data.image_size)
            perturbed_samples_z = perturbed_samples_vec[:, 0]
            if self._config.model.class_cond:
                net_x, net_z = net(perturbed_samples_x, perturbed_samples_z, cond_x[:self._config.training.small_batch_size])
            else:
                net_x, net_z = net(perturbed_samples_x, perturbed_samples_z)
            net_x = net_x.view(net_x.shape[0], -1)
            # Predicted N+1-dimensional Poisson field
            pred = torch.cat([net_z[:, None], net_x], dim=1)
             
            loss = torch.mean((field - pred)**2)
 
            loss.backward()
            optimize_fn(optimizer, net , step=state['step'], config=self._config)
            state['step'] += 1
            state['ema'].update(net.parameters())
            
            
            if kwargs.get(("wandb",False)):
                wandb.log({"loss train":loss.item()},step=step)
 

            if step % self._config.training.eval_freq == 0:
                
                if self._config.experiment == 'generation':
                    try:
                        if self._config.model.class_cond:
                            batch_x,cond_x =  next(eval_iter)
                            cond_x =cond_x.to(self._config.device) 
                    except StopIteration:
                        print('stop')
                    else:
                        eval_iter = iter(eval_loader)
                        if self._config.model.class_cond:
                            batch_x,cond_x =  next(eval_iter)
                            cond_x =cond_x.to(self._config.device)
                        else:
                            batch_x,_ =  next(eval_iter)
                    batch_x = batch_x.to(self._config.device) 
                    batch_y = torch.randn_like(batch_x).to(self._config.device)
                    
                elif self._config.experiment == 'translation':
                    
                    for idx, name_set in enumerate(self._config.data.name_sets):
                        try:
                            batch,_ =  next(eval_iter[name_set])
                        except StopIteration:
                            print('stop')
                        else:
                            eval_iter[name_set] = iter(eval_loader[name_set])
                            batch,_ =  next(eval_iter[name_set])
                        batch = batch.to(self._config.device)

                        if idx == 0:
                            batch_x = batch
                        else:
                            batch_y = batch
                else:
                    raise ValueError

                if self._config.training.plan_type == "Optimal":
                
                    ot_emd = ot.da.EMDTransport(metric = 'euclidean')
                    dot = ot_emd.fit(Xs = batch_x.view(-1, self._config.DIM-1).cpu().numpy(),
                                     Xt = batch_y.view(-1, self._config.DIM-1).cpu().numpy() )
                    batch_y = torch.from_numpy(ot_emd.transform(Xs=batch_x.view(-1, 
                              self._config.DIM-1).cpu().numpy())).to(self._config.device)

                elif self._config.training.plan_type == "Independent":
                    pass
                else:
                    raise ValueError

                batch_y = batch_y.view(-1, self._config.data.num_channels,
                                           self._config.data.image_size,self._config.data.image_size)
                batch_x = batch_x.view(-1, self._config.data.num_channels,
                                       self._config.data.image_size,self._config.data.image_size)
                
                
                with torch.no_grad():
                    ema = state['ema']
                    ema.store(net.parameters())
                    ema.copy_to(net.parameters())
                    
                    perturbed_samples_vec = self.forward_interpolation(batch_x[:self._config.training.small_batch_size],
                                                                       batch_y[:self._config.training.small_batch_size])
            
                    field = self.GroundTruth(perturbed_samples_vec,
                                             torch.cat([self._config.p.x_loc*\
                                                          torch.ones(len(batch_x))[:,None].to(self._config.device),
                                                          batch_x.view(-1,self._config.DIM-1)], dim=1),
                                             torch.cat([self._config.q.x_loc*\
                                                          torch.ones(len(batch_y))[:,None].to(self._config.device),
                                                          batch_y.view(-1,self._config.DIM-1)], dim=1))
                    
                    field = math.sqrt(self._config.DIM)*field/( torch.norm(field, dim=1, keepdim=True) + 1e-5) 
                    perturbed_samples_x = perturbed_samples_vec[:, 1:].view(-1,self._config.data.num_channels,
                                                                    self._config.data.image_size,
                                                                    self._config.data.image_size)
                    perturbed_samples_z = perturbed_samples_vec[:, 0]
                    #net_x, net_z = net(perturbed_samples_x, perturbed_samples_z)
                    if self._config.model.class_cond:
                        net_x, net_z = net(perturbed_samples_x, perturbed_samples_z, 
                                           cond_x[:self._config.training.small_batch_size])
                    else:
                        net_x, net_z = net(perturbed_samples_x, perturbed_samples_z)
                        
                    net_x = net_x.view(net_x.shape[0], -1)
                    # Predicted N+1-dimensional Poisson field
                    pred = torch.cat([net_z[:, None], net_x], dim=1)

                    eval_loss = torch.mean((field - pred)**2)
   
                    ema.restore(net.parameters())
                    if kwargs.get(("wandb",False)):
                        wandb.log({"loss eval":eval_loss.item()},step=step)
                    
            
            # sampling #
            
            if step % self._config.training.snapshot_freq == 0:
                
                with torch.no_grad():
                    ema.store(net.parameters())
                    ema.copy_to(net.parameters())

                    shape = (25, self._config.data.num_channels,
                                 self._config.data.image_size, self._config.data.image_size)
                    
                    if self._config.experiment == 'generation':
                        batch_y = torch.randn(*shape).to(self._config.device)
                    else:
                        batch,_ = next(eval_iter[self._config.data.name_sets[-1]])
                        batch_y = batch[:25].clone()
                    
                    # first sampling procedure #
                    """ 
                    sampling_fn = get_rk45_sampler_pfgm(y=batch_y , config=self._config,
                                                       shape=shape,
                                                       eps=self._config.training.epsilon,
                                                       device=self._config.device)
                    sample, n, traj = sampling_fn(net, batch_y)
                    # first sampling procedure #
                    
                    sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
                    batch_y = np.clip(batch_y.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
                    fig_1 = self.plot(sample.reshape(5,5,self._config.data.image_size,self._config.data.image_size,3) )
                    fig_2 = self.plot(batch_y.reshape(5,5,self._config.data.image_size,self._config.data.image_size,3) )
                    fig_3 = self.plot_trajectory(traj)
                    wandb.log({"Generated Images RK45":fig_1},step=step)
                    wandb.log({"Init Images":fig_2},step=step)
                    wandb.log({"Trajectories RK45":fig_3},step=step)
                    """

                   
                    # second sampling procedure #
                    ode_solver = LearnedImageODESolver(net , self._config)
                    
                    if self._config.experiment == 'generation':
                        batch_y = torch.randn(*shape).to(self._config.device)
                    else:
                        #batch_y,_ = next(eval_iter[self._config.data.name_sets[-1]])
                        batch_y = batch[:25].clone().to(self._config.device)
                        
                    
                    sample, traj = ode_solver(torch.cat([(self._config.L)*torch.ones(batch_y.shape[0],
                                                                                    device=batch_y.device)[:,None],
                                                         batch_y.view(-1, self._config.DIM-1)],dim=1).to(self._config.device))
                    # second sampling procedure #
                    
                    #sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
                    #batch_y = np.clip(batch_y.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
                   
                    fig_1 = self.plota(sample[:,1:].reshape(5,5,3,self._config.data.image_size,
                                                            self._config.data.image_size).detach().cpu() )
                    
                    fig_1.savefig(f"/trinity/home/a.kolesov/EFM/pics/\
{self._config.experiment}/{self._config.data.name}/{self._config.name_exp}.png")
                     
                    
                    #fig_3 = self.plot_trajectory(traj)
                    if kwargs.get(("wandb",False)):
                        wandb.log({"Generated Images Euler":fig_1},step=step)
                    #wandb.log({"Trajectories Euler":fig_3},step=step)
                    
                    torch.save(net.cpu().state_dict(),f"/trinity/home/a.kolesov/EFM/ckpt/{self._config.experiment}/{self._config.data.name}/{self._config.name_exp}.pth")
                    net.to(self._config.device)
    
        return net, state
    #######################################