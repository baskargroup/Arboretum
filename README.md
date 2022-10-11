# VL Hub[](#vl-hub)

**VL Hub** integrates [CLIP pretraining, LiT-Tuning](https://github.com/mlfoundations/open_clip), [alternate CLIP-like architectures](https://github.com/lucidrains/x-clip), [CoCa](https://github.com/lucidrains/coca-pytorch),  conventional [timm](https://github.com/rwightman/pytorch-image-models) vision models and [SimCLR](https://github.com/facebookresearch/SLIP/blob/main/LICENSE) contrastive models into a single test-train-eval framework, making it easier to compare models across architectures and objectives.

VL Hub offers out-of-the-box support for a wide range of zero-shot evaluation metrics, including ImageNet and its distribution shifts, food101, fgvc-Aircraft, and iNaturalist.

## Attribution

This software is heavily indebted to all of the above listed packages, and particularly to [open clip](https://github.com/mlfoundations/open_clip).

## How to use this repository

If you're new to CLIP, you might want to familiarize yourself with the basics of the architecture.

If you haven't worked with the OpenCLIP implementation of CLIP before, the best place to start is the OpenCLIP readme in the docs folder.

If you're planning to use one of the alternate architectures for training, we recommend using the links [above](#vl-hub) to familiarize yourself with the details of that architecture before proceeding.

In this readme, we focus on features that are new in our implementation.

## Gradient Caching[](#gradient-caching)

VL Hub offers support for gradient caching, as described in [GradCache](https://github.com/luyug/GradCache).

Models like CLIP are typically trained with very large batch sizes -- 32,768 is standard. This has been shown to improve rates of loss convergence. 

However, most researchers and even most businesses do not have access to the distributed training setups required for such batch sizes. Gradient caching saves computed gradients in RAM instead of in VRAM, allowing for much larger batch sizes on a single node.

Unlike gradient accumulation, gradient caching is mathematically identical to training with a large batch size. 

To try gradient caching, use these three new command line arguments --

--gc=True: If set to True, gradient caching will be enabled. If gc=True, you should also set gpumaxbatch to an appropriate size for your GPU.

--gpumaxbatch=INT: This is an integer value representing the maximum batch size that your GPU can handle 

--no_eval=True: Skips the evaluation phase, accelerating training