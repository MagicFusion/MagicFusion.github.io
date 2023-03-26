from safetensors import safe_open
from safetensors.torch import save_file


import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid, save_image
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def load_model_from_config_cartoon(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    if ckpt[-4:]!='ckpt':
        sd = {}
        with safe_open(ckpt, framework="pt", device="cpu") as f:
            for key in f.keys():
                sd[key] = f.get_tensor(key)
    else:

        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=int,
        default=0,
        help="mode 0,1,2"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--prompt1",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=100,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=8,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=10.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--model0",
        type=str,
        default="sd-v1-4-full-ema.ckpt",  # stabel diffusion
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--model1",
        type=str,
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=6854,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )


    parser.add_argument(
        "--embedding_path", 
        type=str, 
        help="Path to a pre-trained embedding manager checkpoint")

    parser.add_argument(
        "--fusion_selection",
        type=str,
        default='0')

    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"



    config = OmegaConf.load(f"{opt.config}")

    model0 = load_model_from_config(config, f"{opt.model0}")   #stable diffusion
    model1 = load_model_from_config_cartoon(config, f"{opt.model1}")  #cartoon model

    expert_model = model1
    general_model= model0

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    general_model = general_model.to(device)
    expert_model = expert_model.to(device)

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

    def magicfusion(fusion_selection,seed,model,model0=None,prompt=None,prompt1=None,out_name="fusion",merge_mode=1):
        sampler = DDIMSampler(model,merge_mode=merge_mode,model0=model0)
        assert prompt is not None
        if prompt1 is None:
            data = [batch_size * [prompt]]
        else:
            data = [batch_size * [prompt] + batch_size * [prompt1]]
        seed_everything(seed)
        file_name = f"seed_{seed}_{fusion_selection.replace(',', '_')}"
        sample_path = os.path.join(opt.outdir, f'{out_name}_{merge_mode}')
        os.makedirs(sample_path, exist_ok=True)
        mykwargs = {"fusion_selection": fusion_selection}
        base_count = len(os.listdir(sample_path))
        # grid_count = len(os.listdir(outpath)) - 1

        start_code = None
        if opt.fixed_code:
            start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

        precision_scope = autocast if opt.precision == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    all_samples = list()
                    for n in trange(opt.n_iter, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if opt.scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)
                            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                            samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                             conditioning=c,
                                                             batch_size=opt.n_samples,
                                                             shape=shape,
                                                             verbose=False,
                                                             unconditional_guidance_scale=opt.scale,
                                                             unconditional_conditioning=uc,
                                                             eta=opt.ddim_eta,
                                                             x_T=start_code,
                                                             ref_img=None,
                                                             mykwargs=mykwargs)

                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                            if not opt.skip_save:
                                for x_sample in x_samples_ddim:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    os.makedirs(os.path.join(sample_path, f'samples_{file_name}'), exist_ok=True)
                                    Image.fromarray(x_sample.astype(np.uint8)).save(
                                        os.path.join(sample_path, f'samples_{file_name}', f"{base_count:05}.jpg"))
                                    base_count += 1

                            if not opt.skip_grid:
                                all_samples.append(x_samples_ddim)

                    if not opt.skip_grid:
                        # additionally, save as grid
                        grid = torch.stack(all_samples, 0)
                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                        grid = make_grid(grid, nrow=n_rows)

                        # to image
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        Image.fromarray(grid.astype(np.uint8)).save(os.path.join(sample_path, f'{file_name}.jpg'))
                        # grid_count += 1

        print(f"Your samples are ready and waiting for you here: \n{sample_path} \n")

    seed_list = range(6854, 10000, 564)[:1]   # You can set any seed to explore
    for seed in seed_list:
        # Noise generated by 'model' are used when MagicFusion is disabled.
        magicfusion('0',seed,model=general_model,prompt=opt.prompt1,out_name='general_model',merge_mode=0) # generate image for general model with prompt=prompt1
        magicfusion('0',seed,model=expert_model, prompt=opt.prompt,out_name='expert_model',merge_mode=0) # generate image for expert model with prompt=prompt
        magicfusion(opt.fusion_selection,seed,model=general_model,model0=expert_model,prompt=opt.prompt,prompt1=opt.prompt1,out_name='fusion_ge',merge_mode=opt.mode)
        # For prompt "a bee is making honey", you should exchange the value of 'model' and 'model0', i.e., model=expert_model, and model0=general_model.
        # magicfusion(opt.fusion_selection, seed, model=expert_model, model0=general_model, prompt=opt.prompt, prompt1=opt.prompt1, out_name='fusion_eg', merge_mode=opt.mode)


if __name__ == "__main__":
    main()
