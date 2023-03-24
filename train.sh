CUDA_VISIBLE_DEVICES=5,6 python main.py --base configs/stable-diffusion/v1-finetune_unfrozen.yaml -t --actual_resume ./sd-v1-4-full-ema.ckpt -n gucci --gpus 0,1 --data_root img/gucci/ --reg_data_root outputs/txt2img-samples/samples/ --class_word bag --no-test True

