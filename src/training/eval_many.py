import subprocess
from tqdm.auto import tqdm
import os

def image_size_set(model_name):
    if "384" in model_name:
        image_size = 384
    elif "512" in model_name:
        image_size = 512
    elif "256" in model_name:
        image_size = 256
    elif "336" in model_name:
        image_size = 336
    elif "280" in model_name:
        image_size = 280
    elif "475" in model_name:
        image_size = 475
    elif "448" in model_name:
        image_size = 448
    else:
        image_size = 224
    return image_size

models = [
 'vit_base_patch32_384.augreg_in1k',
 'vit_base_patch16_224.augreg_in1k',
 'vit_base_patch16_384.augreg_in1k',
 'vit_base_patch32_224.sam_in1k',
 'vit_base_patch16_224.sam_in1k',
 'vit_small_patch16_224.dino',
 'vit_small_patch8_224.dino',
 'vit_base_patch16_224.dino',
 'vit_base_patch8_224.dino',
 'vit_small_patch14_dinov2.lvd142m',
 'vit_base_patch14_dinov2.lvd142m',
 'vit_large_patch14_dinov2.lvd142m',
 'vit_giant_patch14_dinov2.lvd142m',
 'vit_base_patch16_224_miil.in21k_ft_in1k',
 'vit_base_patch16_rpn_224.sw_in1k',
 'vit_medium_patch16_gap_256.sw_in12k_ft_in1k',
 'vit_medium_patch16_gap_384.sw_in12k_ft_in1k',
 'vit_base_patch16_gap_224',
 'vit_base_patch32_clip_224.laion2b_ft_in12k_in1k',
 'vit_base_patch32_clip_384.laion2b_ft_in12k_in1k',
 'vit_base_patch32_clip_448.laion2b_ft_in12k_in1k',
 'vit_base_patch16_clip_224.laion2b_ft_in12k_in1k',
 'vit_base_patch16_clip_384.laion2b_ft_in12k_in1k',
 'vit_large_patch14_clip_224.laion2b_ft_in12k_in1k',
 'vit_large_patch14_clip_336.laion2b_ft_in12k_in1k',
 'vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k',
 'vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k',
 'vit_base_patch32_clip_224.openai_ft_in12k_in1k',
 'vit_base_patch32_clip_384.openai_ft_in12k_in1k',
 'vit_base_patch16_clip_224.openai_ft_in12k_in1k',
 'vit_base_patch16_clip_384.openai_ft_in12k_in1k',
 'vit_large_patch14_clip_224.openai_ft_in12k_in1k',
 'vit_large_patch14_clip_336.openai_ft_in12k_in1k',
 'vit_base_patch32_clip_224.laion2b_ft_in1k',
 'vit_base_patch16_clip_224.laion2b_ft_in1k',
 'vit_base_patch16_clip_384.laion2b_ft_in1k',
 'vit_large_patch14_clip_224.laion2b_ft_in1k',
 'vit_large_patch14_clip_336.laion2b_ft_in1k',
 'vit_huge_patch14_clip_224.laion2b_ft_in1k',
 'vit_huge_patch14_clip_336.laion2b_ft_in1k',
 'vit_base_patch32_clip_224.openai_ft_in1k',
 'vit_base_patch16_clip_224.openai_ft_in1k',
 'vit_base_patch16_clip_384.openai_ft_in1k',
 'vit_large_patch14_clip_224.openai_ft_in1k',
 'eva_large_patch14_196.in22k_ft_in22k_in1k',
 'eva_large_patch14_336.in22k_ft_in22k_in1k',
 'eva_large_patch14_196.in22k_ft_in1k',
 'eva_large_patch14_336.in22k_ft_in1k',
 'flexivit_small.1200ep_in1k',
 'flexivit_small.600ep_in1k',
 'flexivit_small.300ep_in1k',
 'flexivit_base.1200ep_in1k',
 'flexivit_base.600ep_in1k',
 'flexivit_base.300ep_in1k',
 'flexivit_large.1200ep_in1k',
 'flexivit_large.600ep_in1k',
 'flexivit_large.300ep_in1k',
 'vit_huge_patch14_224_ijepa.in1k',
 'vit_huge_patch16_448_ijepa.in1k']

caption_subsets = ["--caption-subset=in100", "--caption-subset=in100_dogs", ""]

save_csv = "/scratch/bf996/vlhub/logs/imagenet_timm_sweep.csv"

increment = 0

while os.path.exists(save_csv):
    increment += 1
    save_csv = "/scratch/bf996/vlhub/logs/imagenet_timm_sweep_{}.csv".format(increment)


script_path = 'python src/training/main.py --batch-size=128 --workers=8 --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --model={} --zeroshot-frequency=1 --linear-probe=True {} --image-size={} --save-results-to-csv={};'

pythonpath_cmd = 'export PYTHONPATH="$PYTHONPATH:/scratch/bf996/vlhub/src"'

subprocess.run(pythonpath_cmd, shell=True)

for model in tqdm(models):
    for subset in caption_subsets:
        image_size = str(image_size_set(model))
        command = script_path.format(model, subset, image_size, save_csv)
        try:
            subprocess.run(command, shell=True)
        except Exception as e:
            print(e)
            continue