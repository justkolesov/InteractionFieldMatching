import torch

from tqdm import tqdm
import math
import wandb 

import sys
sys.path.append("/trinity/home/a.kolesov/EFM")
from src.utils import forward_interpolation
from src.ode import LearnedImageODESolver

import matplotlib.pyplot as plt

#########################
def ToyTrain(func, p_dist, q_dist, net, optimizer, config):


    losses = []
    for step in tqdm(range(config.model.training_steps)):

        optimizer.zero_grad()
        x =  p_dist.sample(config.model.training_batch).to(config.device) #[B,DIM]
        y =  q_dist.sample(config.model.training_batch).to(config.device)
        
       
        perturbed_samples_vec = forward_interpolation(x,y,config)

        field = func(perturbed_samples_vec, x.clone(), y.clone(), config) 
        field = math.sqrt(config.DIM)*field/( torch.norm(field, dim=1, keepdim=True) + 1e-5)
        pred  = net(perturbed_samples_vec)
        loss = torch.mean((field - pred)**2)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
    return net, losses
#########################


def Train(func, net, optimizer,  optimize_fn, state, train_loader, eval_loader, config):
    
    train_iter = iter(train_loader)
    eval_iter = iter(eval_loader)
    
    for step in tqdm(range(config.training.n_iters)):

        optimizer.zero_grad()

        try:
            batch_x,_ = next(train_iter)
        except:
            print('Stop Iteration')
        else:
            train_iter = iter(train_loader)
            batch_x,_ = next(train_iter)
        batch_x = batch_x.to(config.device)

        batch_y = torch.randn_like(batch_x).to(config.device)

        stacked_dim = config.p.x_loc*torch.ones(batch_x.shape[0])[:,None].to(config.device)
        batch_x = torch.cat([stacked_dim,batch_x.view(-1,config.DIM-1)], dim=1)

        stacked_dim = config.q.x_loc*torch.ones(batch_y.shape[0])[:,None].to(config.device)
        batch_y = torch.cat([stacked_dim,batch_y.view(-1,config.DIM-1)], dim=1)


        perturbed_samples_vec = forward_interpolation(batch_x ,batch_y , config)
        try:
            assert perturbed_samples_vec.shape[0] == config.training.small_batch_size
        except:
            print("Assert")
            print(perturbed_samples_vec.shape[0])


        field = func(perturbed_samples_vec, batch_x.clone(), batch_y.clone(), config) 
        field = math.sqrt(config.DIM)*field/( torch.norm(field, dim=1, keepdim=True) + 1e-5)

        perturbed_samples_x = perturbed_samples_vec[:, 1:].view(config.training.small_batch_size,-1)
        perturbed_samples_z =  perturbed_samples_vec[:, 0]
        net_x, net_z = net(perturbed_samples_x.view(-1,config.data.num_channels,
                                                    config.data.image_size,
                                                    config.data.image_size), perturbed_samples_z.view(-1))
        pred = torch.cat([net_z[:, None], net_x.view(-1,config.DIM-1)], dim=1)

        loss = torch.mean((field - pred)**2)
        loss.backward()
        optimize_fn(optimizer, net, step, config, config.optim.lr,
                          config.optim.warmup,
                          config.optim.grad_clip)
        state['step'] += 1
        state['ema'].update(net.parameters())
        optimizer = state['optimizer']

        wandb.log({"loss train":loss.item()},step=step)

        if step % config.training.eval_freq == 0:

            try:
                batch_x,_ = next(eval_iter)
            except:
                print('Stop Iteration')
            else:
                eval_iter = iter(eval_loader)
                batch_x,_ = next(eval_iter)
            batch_x = batch_x.to(config.device)

            batch_y = torch.randn_like(batch_x).to(config.device)


            stacked_dim = config.p.x_loc*torch.ones(batch_x.shape[0])[:,None].to(config.device)
            batch_x = torch.cat([stacked_dim,batch_x.view(-1,config.DIM-1)], dim=1)

            stacked_dim = config.q.x_loc*torch.ones(batch_y.shape[0])[:,None].to(config.device)
            batch_y = torch.cat([stacked_dim,batch_y.view(-1,config.DIM-1)], dim=1)


            with torch.no_grad():
                ema = state['ema']
                ema.store(net.parameters())
                ema.copy_to(net.parameters())

                perturbed_samples_vec = forward_interpolation(batch_x ,
                                                              batch_y , config)

                field = func(perturbed_samples_vec, batch_x.clone(), batch_y.clone(), config) 
                field = math.sqrt(config.DIM)*field/( torch.norm(field, dim=1, keepdim=True) + 1e-5)

                perturbed_samples_x = perturbed_samples_vec[:, 1:].view(config.training.small_batch_size,-1)

                net_x, net_z = net(perturbed_samples_x.view(-1,config.data.num_channels,
                                                            config.data.image_size,
                                                            config.data.image_size), perturbed_samples_z.view(-1 ))
                pred = torch.cat([net_z[:, None], net_x.view(-1,config.DIM-1)], dim=1)

                eval_loss = torch.mean((field - pred)**2)

                ema.restore(net.parameters())
                wandb.log({"loss eval":eval_loss.item()},step=step)


        if step % config.training.snapshot_freq == 0:

            with torch.no_grad():
                ema.store(net.parameters())
                ema.copy_to(net.parameters())

                # backward ode setting;
                ode = LearnedImageODESolver(net, config)
                stacked_dim = config.q.x_loc*torch.ones(16)[:,None].to(config.device)
                x_eval = torch.randn(16,config.data.num_channels,
                                     config.data.image_size,
                                     config.data.image_size).view(-1,config.DIM-1) 
                s = torch.cat([stacked_dim, x_eval.to(config.device)], dim=1) 
                s[:,0] = config.q.x_loc - config.training.epsilon
                mapped,traj = ode(s.clone())
                ema.restore(net.parameters())

                fig,ax = plt.subplots(2,16,figsize=(16,2))
                for idx in range(16):
                    ax[0,idx].imshow(traj[0][idx].permute(1,2,0))
                    ax[1,idx].imshow(traj[-2][idx].permute(1,2,0))
                    ax[0,idx].set_yticks([]);ax[0,idx].set_xticks([]);
                    ax[1,idx].set_yticks([]);ax[1,idx].set_xticks([]);
                fig.tight_layout(pad=0.001)

                wandb.log({"Generated Images":fig},step=step)
                
                torch.save(net.cpu().state_dict(), 
                          f"/trinity/home/a.kolesov/EFM/ckpt/ImageGeneration/{config.data.name}/{config.name_exp}.pth")
                net = net.to(config.device)
                
                
                
    return net, state