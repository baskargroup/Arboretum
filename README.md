

# Arboretum: A Large Multimodal Dataset Enabling AI for Biodiversity
## [Project page](https://baskargroup.github.io/Arboretum/)


### Academic Project Page Template
This is an academic paper project page template.


Example project pages built using this template are:
- https://vision.huji.ac.il/spectral_detuning/
- https://vision.huji.ac.il/podd/
- https://dreamix-video-editing.github.io
- https://vision.huji.ac.il/conffusion/
- https://vision.huji.ac.il/3d_ads/
- https://vision.huji.ac.il/ssrl_ad/
- https://vision.huji.ac.il/deepsim/



## Start using the template
To start using the template click on `Use this Template`.

The template uses html for controlling the content and css for controlling the style. 
To edit the websites contents edit the `index.html` file. It contains different HTML "building blocks", use whichever ones you need and comment out the rest.  

**IMPORTANT!** Make sure to replace the `favicon.ico` under `static/images/` with one of your own, otherwise your favicon is going to be a dreambooth image of me.

## Components
- Teaser video
- Images Carousel
- Youtube embedding
- Video Carousel
- PDF Poster
- Bibtex citation

## Model validation

For validating the zero-shot accuracy of our trained models and comparing to other benchmarks, we use the [VLHub](https://github.com/penfever/vlhub) repository with some slight modifications.

### Pre-Run

After cloning this repository and navigating to the `Arboretum/model_validation` directory, we recommend installing all the project requirements into a conda container; `pip install -r requirements.txt`. Also, before executing a command in VLHub, please add `Arboretum/model_validation/src` to your PYTHONPATH.

```bash
export PYTHONPATH="$PYTHONPATH:$PWD/src";
```

### Base Command

A basic Arboretum model evaluation command can be launched as follows:

```bash
python src/training/main.py --batch-size=32 --workers=8 --imagenet-val "/imagenet/val/" --model="resnet50" --zeroshot-frequency=1 --linear-probe=True --image-size=224 --resume "/PATH/TO/WEIGHTS.pth" --report-to wandb
```

To use the open_clip model, you have to first run the `model_validation/load_openclip.py` script to get the model weights from Huggingface and save them locally. The Birds 525 dataset can be downloaded here [here](https://www.kaggle.com/datasets/gpiosenka/100-bird-species) and the IP102 Insects dataset can be found [here](https://drive.google.com/drive/folders/1svFSy2Da3cVMvekBwe13mzyx38XZ9xWo). Once you've downloaded the dataset you can reproduce our model evaluations by running `model_validation/src/training/main.py` with specifying the dataset with the `--ds-filter` argument.

## Tips:
- The `index.html` file contains comments instructing you what to replace, you should follow these comments.
- The `meta` tags in the `index.html` file are used to provide metadata about your paper 
(e.g. helping search engine index the website, showing a preview image when sharing the website, etc.)
- The resolution of images and videos can usually be around 1920-2048, there rarely a need for better resolution that take longer to load. 
- All the images and videos you use should be compressed to allow for fast loading of the website (and thus better indexing by search engines). For images, you can use [TinyPNG](https://tinypng.com), for videos you can need to find the tradeoff between size and quality.
- When using large video files (larger than 10MB), it's better to use youtube for hosting the video as serving the video from the website can take time.
- Using a tracker can help you analyze the traffic and see where users came from. [statcounter](https://statcounter.com) is a free, easy to use tracker that takes under 5 minutes to set up. 
- This project page can also be made into a github pages website.
- Replace the favicon to one of your choosing (the default one is of the Hebrew University). 
- Suggestions, improvements and comments are welcome, simply open an issue or contact me. You can find my contact information at [https://pages.cs.huji.ac.il/eliahu-horwitz/](https://pages.cs.huji.ac.il/eliahu-horwitz/)

## Acknowledgments

If you find this repository useful, please consider citing these related papers --

VLHub

```bibtex
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

BioCLIP

```bibtex
@misc{stevens2024bioclip,
      title={BioCLIP: A Vision Foundation Model for the Tree of Life}, 
      author={Samuel Stevens and Jiaman Wu and Matthew J Thompson and Elizabeth G Campolongo and Chan Hee Song and David Edward Carlyn and Li Dong and Wasila M Dahdul and Charles Stewart and Tanya Berger-Wolf and Wei-Lun Chao and Yu Su},
      year={2024},
      eprint={2311.18803},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

OpenCLIP

```bibtex
@software{ilharco_gabriel_2021_5143773,
  author       = {Ilharco, Gabriel and
                  Wortsman, Mitchell and
                  Wightman, Ross and
                  Gordon, Cade and
                  Carlini, Nicholas and
                  Taori, Rohan and
                  Dave, Achal and
                  Shankar, Vaishaal and
                  Namkoong, Hongseok and
                  Miller, John and
                  Hajishirzi, Hannaneh and
                  Farhadi, Ali and
                  Schmidt, Ludwig},
  title        = {OpenCLIP},
  month        = jul,
  year         = 2021,
  note         = {If you use this software, please cite it as below.},
  publisher    = {Zenodo},
  version      = {0.1},
  doi          = {10.5281/zenodo.5143773},
  url          = {https://doi.org/10.5281/zenodo.5143773}
}
```

Parts of this project page were adopted from the [Nerfies](https://nerfies.github.io/) page.

## Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
