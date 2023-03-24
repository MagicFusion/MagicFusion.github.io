"""SAMPLING ONLY."""
from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor

from torchvision.transforms import Resize
import torch.nn as nn

resize1 = nn.AvgPool2d(4, padding=1, stride=1)
resize2 = Resize(64)
soft = nn.Softmax(dim=1)
import os
from PIL import Image
from einops import rearrange

from sklearn.preprocessing import StandardScaler
#scales_ss = StandardScaler()
from matplotlib import pyplot as plt
ceps0_avg = torch.empty(2,8,4,64,64)
isfirst=1

from PIL import Image 
def read_img(path,batch=1):
    new_img = Image.open(path)
    new_img.load()
    b = np.array(new_img)
    print("b.shape",b.shape)
    pil_image = new_img
    pil_image = pil_image.convert("RGB")
    pil_image = pil_image.resize((64, 64), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    image = arr /255
    image = np.transpose(image, [2, 0, 1])
    image = torch.tensor(image)#.type(th.float16)
    new_img = image.reshape((1,3,64,64))
    new_img = new_img[:,0,:,:]

    new_img = new_img.repeat(batch,4,1,1)

def get_mask(mask_path):
    spe_mask = read_img(mask_path,8)
    spe_mask = spe_mask.unsqueeze(0)
    spe_mask= torch.LongTensor(spe_mask.long())
    spe_mask = spe_mask.to('cuda')
    return spe_mask

def draw_altitude(data,save_path,figure_name):  # 64,64
    plt.rcParams['axes.unicode_minus'] = False
    y, x = np.mgrid[64:0:64j,0:64:64j]
    z = data.cpu().numpy()
    show_both = 0
    if show_both:
        fig = plt.figure(figsize=(10,4))
        ax1 = fig.add_subplot(121)
        ax1.set_title('(a)')
        c1 = ax1.contour(x, y, z, levels=16, cmap='jet')
        ax1.clabel(c1, inline=1, fontsize=8)
        ax2 = fig.add_subplot(122)
        ax2.set_title('(b)')
        c2 = ax2.contourf(x, y, z, levels=16, cmap='jet')
        fig.colorbar(c1, ax=ax1)
        fig.colorbar(c2, ax=ax2)
    else:
        fig = plt.figure(figsize=(5,4))
        ax1 = fig.add_subplot(111)
        ax1.set_title(figure_name)
        c1 = ax1.contourf(x,y,z,levels=16,cmap='jet')
        fig.colorbar(c1,ax=ax1)
    plt.savefig(save_path)


def normalization(data):
    data=data.cpu().numpy()
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
    m = data.shape[0]
    normData = data - np.tile(minVals, (m, 1))
    normData = normData/np.tile(ranges, (m, 1))
    normData = torch.from_numpy(normData).to('cuda')
    return normData



class DDIMSampler(object):
    def __init__(self, model, merge_mode=0,model0=None, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.model0 = model0
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.noise = torch.load('xt.pt')#.chunk(2)[0].repeat(2,1,1,1)
        self.merge_mode = merge_mode

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)


    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray], num_inference_steps):
        timestep, next_timestep = min(timestep - self.ddpm_num_timesteps // num_inference_steps, 999), timestep
        alpha_prod_t = self.model.alphas_cumprod[timestep]# if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.model.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample


    @torch.no_grad()
    def ddim_inversion(self, latent):
        all_latent = [latent]
        latent = latent.clone().detach()
        n = len(self.ddim_timesteps)
        #noise_pred = torch.randn(self.noise.shape, device=self.noise.device)
        for i in range(n-1):
            t = self.ddim_timesteps[i+1]
            # noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            noise_pred = self.noise
            latent = self.next_step(noise_pred, t, latent, n)
            all_latent.append(latent)
        return all_latent

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               ref_img=None,
               mykwargs=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                #if cbs != batch_size:
                #    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')
        # if self.merge_mode==2:  # 1 when test car ,2 when dreambooth
        #     x_T = self.ddim_inversion(ref_img)[-1]

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    mykwargs=mykwargs,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,mykwargs=None):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
            #torch.save(img,'xt.pt')
            #img = self.noise#torch.load('xt.pt')
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,mykwargs=mykwargs)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,mykwargs=None):
        b, *_, device = *x.shape, x.device
        

        fusion_selection=mykwargs['fusion_selection']
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            if self.merge_mode == 0:  # without fusion
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, c])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            else:
                fusion_opt = fusion_selection.split(',')
                fusion_start = int(fusion_opt[0])
                ns0 = float(fusion_opt[1])
                ns1 = float(fusion_opt[2])
                c, c0 = c.chunk(2)
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                prompt = torch.cat([unconditional_conditioning, c])
                e_t_uncond, e_e = self.model0.apply_model(x_in, t_in, prompt).chunk(2)  # prompt
                prompt1 = torch.cat([unconditional_conditioning, c0])
                e_t_uncond0, e_g = self.model.apply_model(x_in, t_in, prompt1).chunk(2)  # prompt1
                eps_e = e_t_uncond + unconditional_guidance_scale * (e_e - e_t_uncond)
                eps_g = e_t_uncond0 + unconditional_guidance_scale * (e_g - e_t_uncond0)
                aeps_e = 0 * e_t_uncond0 + unconditional_guidance_scale * (e_e - e_t_uncond0)
                aeps_g = 0 * e_t_uncond0 + unconditional_guidance_scale * (e_g - e_t_uncond0)
                ceps = torch.cat((eps_g.unsqueeze(0), eps_e.unsqueeze(0)), dim=0)
                aceps = torch.cat((aeps_g.unsqueeze(0), aeps_e.unsqueeze(0)), dim=0)
                ceps0 = resize2(resize1((torch.abs(aceps)).reshape((-1, 4, 64, 64)))).reshape(
                    (2, -1, 4, 64, 64)).mean(dim=2, keepdim=True).repeat(1, 1, 4, 1, 1)
                n, c, h, w = ceps0[0].shape
                ceps0[0] = soft(ns0 * ceps0[0].reshape((n * c, h * w))).reshape((n, c, h, w))  # prompt1  # 8,4,64,64
                ceps0[1] = soft(ns1 * ceps0[1].reshape((n * c, h * w))).reshape((n, c, h, w))  # prompt
                argeps = torch.argmax(ceps0, dim=0, keepdim=True)
                if self.merge_mode==1:
                    if t[0]>fusion_start:
                        e_t = eps_g
                    else:  # fusion
                        e_t = torch.gather(ceps,0,argeps)[0]
                elif self.merge_mode==2:
                    if t[0]>fusion_start:
                        e_t = torch.gather(ceps, 0, argeps)[0]
                    else:
                        e_t = eps_g
        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec
