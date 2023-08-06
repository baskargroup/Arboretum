import argparse

import collections, yaml

def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}

def dict_representer(dumper, data):
  return dumper.represent_mapping(_mapping_tag, data.iteritems())

def dict_constructor(loader, node):
  return collections.OrderedDict(loader.construct_pairs(node))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to csv filewith training data",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default=None,
        help="Path to training schema with data and epochs",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to csv file with validation data",
    )
    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Required for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--val-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Useful for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["webdataset", "csv", "synthetic", "imagefolder", "auto"],
        default="auto",
        help="Which type of dataset to process."
    )
    parser.add_argument(
        "--dataset-resampled",
        default=False,
        action="store_true",
        help="Whether to use sampling with replacement for webdataset shard selection."
    )
    parser.add_argument(
        "--size-controlled",
        type=str,
        default="",
        help="Limit the maximum sample and class size of an ImageFolder dataset. Expected format is 'max_sample_size, max_class_size' in integers."
    )   
    parser.add_argument(
        "--csv-separator",
        type=str,
        default="\t",
        help="For csv-like datasets, which separator to use."
    )
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths."
    )
    parser.add_argument(
        "--ds-filter",
        type=str,
        default="",
        help="Filter to only include samples in the batch whose captions contain a word in the filter list: options are ['imagenet_classnames', 'inat_classnames', 'cars_classnames', 'food_classnames', 'air_classnames']."
    )
    parser.add_argument(
        "--strict",
        type=bool,
        default=False,
        help="Strict filtering"
    )
    parser.add_argument(
        "--ideo",
        type=bool,
        default=False,
        help="Use ideogram classnames"
    )
    parser.add_argument(
        "--ds-cipher",
        type=bool,
        default=False,
        help="Filter to only include samples in the batch whose captions contain an ImageNet1k class, and encode those classes using a substitution cipher: caption cleaning required."
    )
    parser.add_argument(
        "--shift-cipher",
        type=int,
        default=None,
        help="Apply a shift cipher to the captions, with a value provided as an argument."
    )
    parser.add_argument(
        "--no-ensembling",
        type=bool,
        default=False,
        help="No prompt ensembling: evaluate on the first prompt in the template"
    )
    parser.add_argument(
        "--simplecaptions",
        type=bool,
        default=False,
        help="Change caption text to 'An image of CLASSNAME, CLASSNAME': class filtering required."
    )
    parser.add_argument(
        "--csv-scrambled",
        type=bool,
        default=False,
        help="Scramble word ordering of captions during training"
    )
    parser.add_argument(
        "--token-scrambled",
        type=bool,
        default=False,
        help="Scramble token ordering of captions during training"
    )
    parser.add_argument(
        "--token-strip",
        type=bool,
        default=False,
        help="Strip tokens not in evaluation dataset during training"
    )
    parser.add_argument(
        "--token-reduce",
        type=bool,
        default=False,
        help="Keep only one non-zero token"
    )
    parser.add_argument(
        "--extended-metrics",
        type=bool,
        default=False,
        help="Confusion matrices and class-wise accuracy"
    )
    parser.add_argument(
        "--save-results-to-csv",
        type=str,
        default="",
        help="Save metrics to csv file"
    )
    parser.add_argument(
        "--zs-upper",
        type=bool,
        default=False,
        help="Force classes to UPPER-CASE during inference"
    )
    parser.add_argument(
        "--zs-lower",
        type=bool,
        default=False,
        help="Force classes to lower-case during inference"
    )
    parser.add_argument(
        "--csv-cleaned",
        type=bool,
        default=False,
        help="Clean captions"
    )
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="title",
        help="For csv-like datasets, the name of the key for the captions."
    )
    parser.add_argument(
        "--imagenet-val",
        type=str,
        default=None,
        help="Path to imagenet val set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-v2",
        type=str,
        default=None,
        help="Path to imagenet v2 for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--objectnet",
        type=str,
        default=None,
        help="Path to objectnet for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--insecta",
        type=str,
        default=None,
        help="Path to insecta for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-s",
        type=str,
        default=None,
        help="Path to imagenet sketch set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-r",
        type=str,
        default=None,
        help="Path to imagenet-r set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-a",
        type=str,
        default=None,
        help="Path to imagenet-a set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--inat2021",
        type=str,
        default=None,
        help="Path to inat 2021 (validation or test folder) for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--inat2018",
        type=str,
        default=None,
        help="Path to inat 2018 (validation or test folder) for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--inat2017",
        type=str,
        default=None,
        help="Path to inat 2017 (validation or test folder) for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--stanfordcars",
        type=str,
        default=None,
        help="Path to stanford cars (validation or test folder) for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-train",
        type=str,
        default=None,
        help="Path to ImageNet training data",
    )
    parser.add_argument(
        "--imagenet-tune-freq",
        type=int,
        default=0,
        help="How many epochs to fine-tune image head on ImageNet while training",
    )
    parser.add_argument(
        "--ramping",
        type=bool,
        default=False,
        help="Ramp up to full dataset length over the course of the training run",
    )
    parser.add_argument(
        "--dry-run",
        type=bool,
        default=False,
        help="Load data but do not actually train model",
    )             
    parser.add_argument(
        "--flowers",
        type=str,
        default=None,
        help="Path to flowers102 (validation or test folder) for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--air",
        type=str,
        default=None,
        help="Path to FGVCAircraft (validation or test folder) for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--food",
        type=str,
        default=None,
        help="Path to food101 (validation or test folder) for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="log files on local master, otherwise global master only.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="vlhub",
        help="Identifier for the wandb project. Default is vlhub."
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=10000, help="Number of steps to warmup for."
    )
    parser.add_argument("--gsam", action="store_true", default=False, help="Use GSAM.")
    parser.add_argument("--clamp", type=float, default=0, help="Gradient clamping.")
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.")
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--zeroshot-frequency", type=int, default=2, help="How often to run zero shot."
    )
    parser.add_argument(
        "--zeroshot-scramble", type=bool, default=False, help="Scramble text ordering for zero-shot inference."
    )
    parser.add_argument(
        "--val-frequency", type=int, default=1, help="How often to run evaluation with val data."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bfloat16", "fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RN50",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--pretrained",
        default='',
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--pretrained-image",
        default=False,
        action='store_true',
        help="Load imagenet pretrained weights for image tower backbone if available.",
    )
    parser.add_argument(
        "--pretrained-head",
        default='',
        type=str,
        help="Replace the vision tower with a fully trained vision model located at the given path.",
    )
    parser.add_argument(
        "--metacaptions",
        default='',
        type=str,
        help="Path to metadata csv for captioning.",
    )
    parser.add_argument(
        "--lock-image",
        default=False,
        action='store_true',
        help="Lock full image tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-text",
        default=False,
        action='store_true',
        help="Lock full text tower by disabling gradients.",
    )
    parser.add_argument(
        "--sim-clr",
        default=False,
        help="Use simclr loss",
    )
    parser.add_argument(
        "--byol",
        default=False,
        help="Use byol loss",
    )
    parser.add_argument(
        "--sim-clr-trans",
        default=False,
        help="Use simclr image transforms",
    )
    parser.add_argument(
        "--augreg-trans",
        default=False,
        action='store_true',
        help="Use simclr image transforms",
    )
    parser.add_argument(
        "--downsample-trans",
        default=False,
        help="Use simclr image transforms with downsampling (jpg quality = 10)",
    )
    parser.add_argument(
        "--add-trunk",
        default=False,
        help="Add the word, trunk, to the model state dict",
    )
    parser.add_argument(
        "--caption-subset",
        default = '',
        type=str,
        help="Run inference only on the classes in the Imagenet-Captioned dataset",
    )
    parser.add_argument(
        "--integer-labels",
        default=False,
        action='store_true',
        help="Train on integer labels instead of text captions.",
    )
    parser.add_argument(
        "--iqe",
        default=False,
        action='store_true',
        help="Use quasimetric embedding distance when computing CLIP loss.",
    )
    parser.add_argument(
        "--alignunif",
        default=False,
        action='store_true',
        help="Use alignment + uniform loss.",
    )
    parser.add_argument(
        "--img-weight", type=float, default=.5, help="How heavily to weight image embedding"
    )
    parser.add_argument(
        "--text-weight", type=float, default=.5, help="How heavily to weight text embedding"
    )
    parser.add_argument(
        "--multiclass",
        default=False,
        action='store_true',
        help="Multiclass integer labels.",
    )
    parser.add_argument(
        "--fine-tune",
        default=False,
        action='store_true',
        help="Resume checkpoint for fine tuning (ignore optimizer and scaler, if they exist)",
    )
    parser.add_argument(
       "--load-first-k",
       type=int,
       default=0,
       help="Load only the first k weights of the model",
    )
    parser.add_argument(
        "--lock-image-unlocked-groups",
        type=int,
        default=0,
        help="Leave last n image tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-image-freeze-bn-stats",
        default=False,
        action='store_true',
        help="Freeze BatchNorm running stats in image tower for any locked layers.",
    )
    parser.add_argument(
        '--image-mean', type=float, nargs='+', default=None, metavar='MEAN',
        help='Override default image mean value of dataset')
    parser.add_argument(
        '--image-std', type=float, nargs='+', default=None, metavar='STD',
        help='Override default image std deviation of of dataset')
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action='store_true',
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--local-loss",
        default=False,
        action="store_true",
        help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)"
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="enable full distributed gradient for feature gather"
    )
    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action='store_true',
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action='store_true',
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action='store_true',
        help="torch.jit.trace the model for inference / eval only",
    )
    # arguments for distributed training
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--report-to",
        default='',
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']"
    )
    parser.add_argument(
        "--wandb-notes",
        default='',
        type=str,
        help="Notes if logging with wandb"
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log diretory, and execute from there."
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training."
    )
    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action='store_true',
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )
    parser.add_argument(
        "--gc",
        default=False,
        help="Use gradient caching",
    )
    parser.add_argument(
        "--gpumaxbatch",
        type=int,
        default=32,
        help="max per-GPU batch size for gradient caching",
    )
    parser.add_argument(
        "--linear-probe",
        default=False,
        help="Linear probing of timm models",
    )
    parser.add_argument(
        "--lsuv",
        default=False,
        action="store_true",
        help="Use LSUV initialization"        
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Resolution of model to probe",
    )
    parser.add_argument(
        "--filip",
        default=False,
        help="whether to use fine-grained contrastive learning (FILIP)",
    )
    parser.add_argument(
        "--dcl",
        default=False,
        help="use decoupled contrastive learning (DCL) objective function, removing positive pairs from the denominator of the InfoNCE loss (CLOOB + DCL)",
    )
    parser.add_argument(
        "--elp",
        default=False,
        help="whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)",
    )
    parser.add_argument(
        "--vssl",
        default=False,
        help="whether to do self supervised learning on images",
    )
    parser.add_argument(
        "--mlm",
        default=False,
        help="use masked language learning (MLM) on text (DeCLIP)",
    )
    parser.add_argument(
        "--text_ssl_loss_weight",
        type=float, default=.05,
        help="use masked language learning (MLM) on text (DeCLIP)",
    )
    parser.add_argument(
        "--image_ssl_loss_weight",
        type=float, default=.05,
        help="use masked language learning (MLM) on text (DeCLIP)",
    )
    parser.add_argument(
        "--norm_gradient_clip", type=float, default=None, help="Gradient clip."
    )
    args = parser.parse_args()

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args
