# VL Hub[](#vl-hub)

**VL Hub** integrates [CLIP pretraining, LiT-Tuning](https://github.com/mlfoundations/open_clip), [alternate CLIP-like architectures](https://github.com/lucidrains/x-clip), [CoCa](https://github.com/lucidrains/coca-pytorch),  conventional [timm](https://github.com/rwightman/pytorch-image-models) vision models and [SimCLR](https://github.com/facebookresearch/SLIP/blob/main/LICENSE) contrastive models into a single test-train-eval framework, making it easier to compare models across architectures and objectives.

## Attribution

This software is heavily indebted to all of the above listed packages, and particularly to [open clip](https://github.com/mlfoundations/open_clip).

If you find this repository useful, please cite the accompanying paper --

```latex
@article{
feuer2023distributionally,
title={Distributionally Robust Classification on a Data Budget},
author={Benjamin Feuer and Ameya Joshi and Minh Pham and Chinmay Hegde},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2023},
url={https://openreview.net/forum?id=D5Z2E8CNsD},
note={}
}
```

## How to use this repository

If you're new to CLIP, you might want to familiarize yourself with the basics of the architecture.

If you haven't worked with the OpenCLIP implementation of CLIP before, the best place to start is the OpenCLIP readme in the docs folder.

If you're in need of a vision-language dataset for your training experiments, check out [CaptionNet](https://github.com/penfever/CaptionNet), a dataset designed precisely for that purpose.

If you're planning to use one of the alternate architectures for training, we recommend using the links [above](#vl-hub) to familiarize yourself with the details of that architecture before proceeding.

In this readme, we focus on features that are new in our implementation.

# Evaluation

## Extended Evaluation Metrics[](#extended-evaluation-metrics)

VL Hub integrates support for a wide range of zero-shot evaluation metrics, including ImageNet and its distribution shifts, food101, fgvc-Aircraft, Stanford Cars, and iNaturalist. Simply add the dataset you wish to evaluate on, and include the following flag(s) when at evaluation time, substituting the appropriate PATH --

--food "/", --air "/", --stanfordcars "/", --imagenet-val "/imagenet/val/", --imagenet-a "/imagenet-a", --imagenet-r "/imagenet-r", --imagenet-v2 "/imagenet-v2", --imagenet-s "/imagenet-sketch", --inat2021 "/inat2021"

## Subset Evaluation[](#subset-evaluation)

ImageNet and its distribution shifts also support evaluation on a 100-class subset of ImageNet-1k; this is particularly useful when training models on smaller datasets such as [CaptionNet](https://github.com/penfever/CaptionNet).

To evaluate on a subset of ImageNet, include the flag --caption-subset=True

## Extended Metrics[](#extended-metrics)

VL Hub supports extended evaluation metrics such as confusion matrices and per-class accuracy results. To utilize these features, pass the flag --extended-metrics=True.

# Training

## Supported Training Architectures

VL Hub currently supports training the following architectures;

* CLIP-loss models

As this is the default training mode, you need only pass the type of model you wish to train, EG, --model=RN50

* LiT-tuned models

Please refer to docs/openclip_readme for details

--lock-image-unlocked-groups will unlock only the last $n$ layers of the image tower during training

--lock-text will lock the text tower during training (not recommended)

* Conventional cross-entropy loss models

In order to train conventional cross-entropy loss models without using a text tower, if your dataset is in CSV format, you must specify a caption-key column which contains either integers or a list of integers.

--csv-caption-key idx

If you are training on webdataset, this step is not necessary, but you should specify a dataset filter --

--ds-filter=imagenet_classnames

Integer labels will be generated using a subset matching strategy. For more details, please see our paper.

In either case, you must also pass

--integer-labels

You should then choose a model architecture with an appropriate head. For instance, if training an ImageNet model, you might choose 

--model=RN50-in1k

* CoCa

In order to train a CoCa model, simply pass

--model=coca

Please note that at this time, support for CoCa is limited to a single vision backbone, and the loss weighting has to be adjusted manually.

* DeCLIP, VSSL, filip

To train using one of these alternate objectives, pass the model architecture you wish to use as your base, and flag the objective you wish to train with. For instance:

--model=vit_base_patch32_224 --mlm=True

* Changing image and text weighting

By passing float values between .1 and .9 to the flags --text-weight and --image-weight, it is possible to change how heavily CLIP weights both text and image loss.

## Training Schema[](#training-schema)

In order to make training on blended or combined datasets more convenient when using webdataset, we implement training schema. Pass the flag

--schema=PATH

and do NOT pass a path to any training data in order to use schema.

Sample schema are provided in the repository.

Please note; schema training is only supported when using webdataset format; if you wish to combine CSV datasets, simply merge the CSVs you wish to combine.

## SimCLR Augmentation[](#simclr-augmentation)

When

--sim-clr-trans=True

is passed, the model will use SimCLR augmentations instead of standard CLIP augmentations. This has been shown to improve model zero-shot performance by as much as 10 percent.

## Gradient Caching[](#gradient-caching)

VL Hub offers support for gradient caching, as described in [GradCache](https://github.com/luyug/GradCache).

Models like CLIP are typically trained with very large batch sizes -- 32,768 is standard. This has been shown to improve rates of loss convergence. 

However, most researchers and even most businesses do not have access to the distributed training setups required for such batch sizes. Gradient caching saves computed gradients in RAM instead of in VRAM, allowing for much larger batch sizes on a single node.

Unlike gradient accumulation, gradient caching is mathematically identical to training with a large batch size. 

To try gradient caching, use these three new command line arguments --

--gc=True: If set to True, gradient caching will be enabled. If gc=True, you should also set gpumaxbatch to an appropriate size for your GPU.

--gpumaxbatch=INT: This is an integer value representing the maximum batch size that your GPU can handle 

--no_eval=True: Skips the evaluation phase, accelerating training

# Subset Matching

The default subset matching strategy is single-class, as this has been shown to perform best. However, other strategies are available. --multiclass enforces multiclass subset matching, whereas --strict utilizes a strict subset matching strategy. For more details on subset matching strategies, please refer to our paper.

# Captioning

VL Hub supports many modifications to captions, both during inference and during training.

"--zs-upper" forces classes to UPPER-CASE during inference, "zs-lower" forces lower-case. "--csv-cleaned" cleans the captions prior to training (this flag also works for webdatasets).

--token-strip strips all tokens not used for evaluation.

--token-scrambled scrambles token order during training.

"--simplecaptions" changes caption text to 'An image of CLASSNAME, CLASSNAME'

--shift-cipher=3 applies a 3-step shift-cipher to the caption space.
