import subprocess
from tqdm.auto import tqdm
import os
import pandas as pd

def image_size_set(model_name, df):
    row = df[df['updated_name'] == model_name]
    if not row.empty:
        imsize = int(row.iloc[0]['image_size'])
        if imsize:
            return imsize
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
 'skresnet18',
 'skresnet34',
 'skresnext50_32x4d',
 'spnasnet_100.rmsp_in1k',
 'ssl_resnet18',
 'ssl_resnet50',
 'ssl_resnext101_32x16d',
 'ssl_resnext101_32x4d',
 'ssl_resnext101_32x8d',
 'ssl_resnext50_32x4d',
 'swin_base_patch4_window12_384',
 'swin_base_patch4_window7_224',
 'swin_large_patch4_window12_384',
 'swin_large_patch4_window7_224',
 'swin_s3_base_224',
 'swin_s3_small_224',
 'swin_s3_tiny_224',
 'swin_small_patch4_window7_224',
 'swin_tiny_patch4_window7_224',
 'swinv2_base_window12to16_192to256_22kft1k',
 'swinv2_base_window12to24_192to384_22kft1k',
 'swinv2_base_window16_256',
 'swinv2_base_window8_256',
 'swinv2_cr_small_224',
 'swinv2_cr_small_ns_224',
 'swinv2_cr_tiny_ns_224',
 'swinv2_large_window12to16_192to256_22kft1k',
 'swinv2_large_window12to24_192to384_22kft1k',
 'swinv2_small_window16_256',
 'swinv2_small_window8_256',
 'swinv2_tiny_window16_256',
 'swinv2_tiny_window8_256',
 'swsl_resnet18',
 'swsl_resnet50',
 'swsl_resnext101_32x16d',
 'swsl_resnext101_32x4d',
 'swsl_resnext101_32x8d',
 'swsl_resnext50_32x4d',
 'tf_efficientnet_b0.aa_in1k',
 'tf_efficientnet_b1.aa_in1k',
 'tf_efficientnet_b2.aa_in1k',
 'tf_efficientnet_b3.aa_in1k',
 'tf_efficientnet_b4.aa_in1k',
 'tf_efficientnet_b5.ra_in1k',
 'tf_efficientnet_b6.aa_in1k',
 'tf_efficientnet_b7.ra_in1k',
 'tf_efficientnet_b8.ap_in1k',
 'tf_efficientnet_cc_b0_4e.in1k',
 'tf_efficientnet_cc_b0_8e.in1k',
 'tf_efficientnet_cc_b1_8e.in1k',
 'tf_efficientnet_el.in1k',
 'tf_efficientnet_em.in1k',
 'tf_efficientnet_es.in1k',
 'tf_efficientnet_l2_ns',
 'tf_efficientnet_lite0.in1k',
 'tf_efficientnet_lite1.in1k',
 'tf_efficientnet_lite2.in1k',
 'tf_efficientnet_lite3.in1k',
 'tf_efficientnet_lite4.in1k',
 'tf_efficientnetv2_b0.in1k',
 'tf_efficientnetv2_b1.in1k',
 'tf_efficientnetv2_b2.in1k',
 'tf_efficientnetv2_b3.in1k',
 'tf_efficientnetv2_l.in1k',
 'tf_efficientnetv2_l.in21k_ft_in1k',
 'tf_efficientnetv2_m.in1k',
 'tf_efficientnetv2_m.in21k_ft_in1k',
 'tf_efficientnetv2_s.in1k',
 'tf_efficientnetv2_s.in21k_ft_in1k',
 'tf_efficientnetv2_xl.in21k_ft_in1k',
 'tf_inception_v3',
 'tf_mixnet_l.in1k',
 'tf_mixnet_m.in1k',
 'tf_mixnet_s.in1k',
 'tf_mobilenetv3_large_075.in1k',
 'tf_mobilenetv3_large_100.in1k',
 'tf_mobilenetv3_large_minimal_100.in1k',
 'tf_mobilenetv3_small_075.in1k',
 'tf_mobilenetv3_small_100.in1k',
 'tf_mobilenetv3_small_minimal_100.in1k',
 'tinynet_a.in1k',
 'tinynet_b.in1k',
 'tinynet_c.in1k',
 'tinynet_d.in1k',
 'tinynet_e.in1k',
 'tnt_s_patch16_224',
 'tv_densenet121',
 'tv_resnet101',
 'tv_resnet152',
 'tv_resnet34',
 'tv_resnet50',
 'tv_resnext50_32x4d',
 'twins_pcpvt_base',
 'twins_pcpvt_large',
 'twins_pcpvt_small',
 'twins_svt_base',
 'twins_svt_large',
 'twins_svt_small',
 'vgg11',
 'vgg11_bn',
 'vgg13',
 'vgg13_bn',
 'vgg16',
 'vgg16_bn',
 'vgg19',
 'vgg19_bn',
 'visformer_small',
 'volo_d1_224',
 'volo_d1_384',
 'volo_d2_224',
 'volo_d2_384',
 'volo_d3_224',
 'volo_d3_448',
 'volo_d4_224',
 'volo_d4_448',
 'volo_d5_224',
 'volo_d5_448',
 'volo_d5_512',
 'wide_resnet101_2',
 'wide_resnet50_2',
 'wide_resnet50_2',
 'xception',
 'xception41',
 'xception41p',
 'xception65',
 'xception65p',
 'xception71',
 'xcit_large_24_p16_224',
 'xcit_large_24_p16_224_dist',
 'xcit_large_24_p16_384_dist',
 'xcit_large_24_p8_224',
 'xcit_large_24_p8_224_dist',
 'xcit_large_24_p8_384_dist',
 'xcit_medium_24_p16_224',
 'xcit_medium_24_p16_224_dist',
 'xcit_medium_24_p16_384_dist',
 'xcit_medium_24_p8_224',
 'xcit_medium_24_p8_224_dist',
 'xcit_medium_24_p8_384_dist',
 'xcit_nano_12_p16_224',
 'xcit_nano_12_p16_224_dist',
 'xcit_nano_12_p16_384_dist',
 'xcit_nano_12_p8_224',
 'xcit_nano_12_p8_224_dist',
 'xcit_nano_12_p8_384_dist',
 'xcit_small_12_p16_224',
 'xcit_small_12_p16_224_dist',
 'xcit_small_12_p16_384_dist',
 'xcit_small_12_p8_224',
 'xcit_small_12_p8_224_dist',
 'xcit_small_12_p8_384_dist',
 'xcit_small_24_p16_224',
 'xcit_small_24_p16_224_dist',
 'xcit_small_24_p16_384_dist',
 'xcit_small_24_p8_224',
 'xcit_small_24_p8_224_dist',
 'xcit_small_24_p8_384_dist',
 'xcit_tiny_12_p16_224',
 'xcit_tiny_12_p16_224_dist',
 'xcit_tiny_12_p16_384_dist',
 'xcit_tiny_12_p8_224',
 'xcit_tiny_12_p8_224_dist',
 'xcit_tiny_12_p8_384_dist',
 'xcit_tiny_24_p16_224',
 'xcit_tiny_24_p16_224_dist',
 'xcit_tiny_24_p16_384_dist',
 'xcit_tiny_24_p8_224',
 'xcit_tiny_24_p8_224_dist',
 'xcit_tiny_24_p8_384_dist',
 ]

df_is = pd.read_csv("/scratch/bf996/vlhub/metadata/meta-analysis-results.csv")

caption_subsets = ["--caption-subset=in100", "--caption-subset=in100_dogs", ""]

save_csv = "/scratch/bf996/vlhub/logs/imagenet_timm_sweep.csv"

increment = 0

while os.path.exists(save_csv):
    increment += 1
    save_csv = "/scratch/bf996/vlhub/logs/imagenet_timm_sweep_{}.csv".format(increment)

script_path = 'python src/training/main.py --batch-size=128 --workers=8 --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --model={} --zeroshot-frequency=1 --linear-probe=True {} --image-size={} --save-results-to-csv={};'

pythonpath_cmd = 'export PYTHONPATH="$PYTHONPATH:/scratch/bf996/vlhub/src"'

subprocess.run(pythonpath_cmd, shell=True)

for idx, model in tqdm(enumerate(models)):
    for subset in caption_subsets:
        image_size = str(image_size_set(model, df_is))
        command = script_path.format(model, subset, image_size, save_csv)
        try:
            subprocess.run(command, shell=True)
        except Exception as e:
            print(e)
            continue