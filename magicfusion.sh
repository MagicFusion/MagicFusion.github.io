
model0="sd-v1-4-full-ema.ckpt"   # stable diffusion model
model1="anything-v3-full.safetensors"    #cartoon model
prompt="A lion with a crown on his head"
prompt1="A lion with a crown on his head"
#prompt="A bee is making honey"
#prompt1="A bee is making honey"

outdir="output/lion_crown"
fusion_selection="600,1,1"
merge_mode=2


CUDA_VISIBLE_DEVICES=4 python scripts/stable_txt2img.py --model0 $model0 --model1 $model1 --outdir $outdir --prompt "$prompt" --prompt1 "$prompt1"  --fusion_selection $fusion_selection --mode $merge_mode
