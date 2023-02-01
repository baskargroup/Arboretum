import logging
import random

import torch
from torch import einsum
import torch.nn.functional as F
from tqdm import tqdm

import numpy as np

from open_clip import tokenize
from .precision import get_autocast
from .imagenet_zeroshot_data import imagenet_classnames, openai_imagenet_template

from .data import shift_cipher

from .imagenet_zeroshot_data import *
from .metrics import *
try:
    from .inat_zeroshot_data import inat_classnames, inat_template
    from .cars_zeroshot_data import cars_classnames, cars_template
    from .food_zeroshot_data import food_classnames, food_template
    from .air_zeroshot_data import air_classnames, air_template
    from .insecta_zeroshot_data import get_insecta_classnames

except Exception as e:
    print(e)

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

def zero_shot_classifier(model, classnames, templates, args):
    logging.debug("In zero-shot-classifer, classnames are {}".format(classnames))
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template(classname) for template in templates]  # format with class
            if args.zeroshot_scramble:
                res = []
                tlist = [t.split(" ") for t in texts]
                for l in tlist:
                    random.shuffle(l)
                    res.append(" ".join(l).strip())
            texts = tokenize(texts).to(args.device)  # tokenize
            logging.debug("In zero-shot-classifer, tokens are {}".format(classnames))
            if args.distributed and not args.horovod:
                if args.model in ["coca"]:
                    images = torch.rand(len(texts), 3, 224, 224).to(args.device)
                    class_embeddings = model.module(texts, images, return_embeddings=True)
                    class_embeddings = class_embeddings[0]
                    class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                    class_embedding /= class_embedding.norm()
                elif args.model in ["xclip"]:
                    images = torch.rand(len(texts), 3, model.module.image_size, model.module.image_size).to(args.device)
                    class_embeddings = model.module(texts, images, return_encodings=True)
                    if args.filip:
                        lat = model.module.to_text_latent(class_embeddings[0][:, 1:]).mean(dim=0)
                        class_embedding = l2norm(lat).mean(dim=0)
                    else:
                        class_embedding = l2norm(model.module.to_text_latent(class_embeddings[0][:, 0])).mean(dim=0)          
                else:
                    class_embeddings = model.module.encode_text(texts)
                    class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                    class_embedding /= class_embedding.norm()
            else:         
                if args.model in ["coca"]:
                    images = torch.rand(len(texts), 3, 224, 224).to(args.device)
                    class_embeddings = model(texts, images, return_embeddings=True)
                    class_embeddings = class_embeddings[0]
                    class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                    class_embedding /= class_embedding.norm()
                elif args.model in ["xclip"]:
                    images = torch.rand(len(texts), 3, model.image_size, model.image_size).to(args.device)
                    class_embeddings = model(texts, images, return_encodings=True)
                    if args.filip:
                        lat = model.to_text_latent(class_embeddings[0][:, 1:]).mean(dim=0)
                        class_embedding = l2norm(lat).mean(dim=0)
                    else:
                        class_embedding = l2norm(model.to_text_latent(class_embeddings[0][:, 0])).mean(dim=0)
                else:
                    class_embeddings = model.encode_text(texts)
                    class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                    class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.device)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args, idx=None, split=None):
    autocast = get_autocast(args.precision)
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        args.y_pred = []
        args.y_true = []
        args.logits = []
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            if args.caption_subset:
                if any([args.integer_labels, args.linear_probe]) and split == "r":
                    ir_idx = get_ir_idx().tolist()
                    match_idx = sum(target==ir_idx.index(i) for i in idx).bool().nonzero(as_tuple=True)[0]
                elif any([args.integer_labels, args.linear_probe]) and split == "a":
                    ia_idx = get_ia_idx().tolist()
                    match_idx = sum(target==ia_idx.index(i) for i in idx).bool().nonzero(as_tuple=True)[0]
                else:
                    match_idx = sum(target==i for i in idx).bool().nonzero(as_tuple=True)[0]
                target = target[match_idx].to(args.device)
                images = images[match_idx].to(args.device)
                if images.size(0) == 0:
                    continue
                if not any([args.integer_labels, args.linear_probe]):
                    idx_l = idx.tolist()
                    target = torch.tensor([idx_l.index(t) for t in target]).to(args.device)
                elif any([args.integer_labels, args.linear_probe]) and split == "r":
                    ir_idx = get_ir_idx()
                    target = torch.tensor(ir_idx[target.cpu()]).to(args.device)
                elif any([args.integer_labels, args.linear_probe]) and split == "a":
                    ia_idx = get_ia_idx()
                    target = torch.tensor(ia_idx[target.cpu()]).to(args.device)             
            else:
                images = images.to(args.device)
                try:
                    s = idx.shape
                    target = target.tolist()
                    target = torch.tensor(idx[target])
                except Exception as e:
                    pass
                target = target.to(args.device)
            #FIXME: handle larger batch sizes gracefully with gradient caching
            if args.gc:
                images = images[:min(args.gpumaxbatch, len(images)-1)]
                target = target[:min(args.gpumaxbatch, len(images)-1)]
            with autocast():
                # predict
                if args.distributed and not args.horovod:
                    if args.linear_probe:
                        logits = model.module(images)
                    elif args.integer_labels:
                        logits = model.module.visual(images)
                    elif args.model == "coca":
                        texts = torch.randint(100, (5, len(images))).to(args.device)
                        image_features = model.module(texts, images, return_embeddings=True)
                        image_features = F.normalize(image_features[1], dim=-1)
                        logits = model.module.temperature.exp() * image_features @ classifier   
                    elif args.model == "xclip":
                        texts = torch.randint(100, (len(images), 5)).to(args.device)
                        image_features = model.module(texts, images, return_encodings=True)
                        if args.filip:
                            image_features = l2norm(model.module.to_visual_latent(image_features[1][:, 1:])).mean(dim=1)
                        else:
                            image_features = l2norm(model.module.to_visual_latent(image_features[1][:, 0]))
                        logits = model.module.temperature.exp() * image_features @ classifier
                    else:
                        image_features = model.module.encode_image(images)
                        image_features = F.normalize(image_features, dim=-1)
                        logits = 100. * image_features @ classifier
                else:
                    if args.linear_probe:
                        logits = model(images)
                    elif args.integer_labels:
                        logits = model.visual(images)
                    elif args.model == "coca":
                        texts = torch.randint(100, (5, len(images))).to(args.device)
                        image_features = model(texts, images, return_embeddings=True)
                        image_features = F.normalize(image_features[1], dim=-1)
                        logits = model.temperature.exp() * image_features @ classifier         
                    elif args.model == "xclip":
                        texts = torch.randint(100, (len(images), 5)).to(args.device)
                        image_features = model(texts, images, return_encodings=True)
                        if args.filip:
                            image_features = l2norm(model.to_visual_latent(image_features[1][:, 1:])).mean(dim=1)
                        else:
                            image_features = l2norm(model.to_visual_latent(image_features[1][:, 0]))
                        #logging.info("size of image_features {}, size of classifier {}".format(image_features.size(), classifier.size()))
                        #FILIP: einsum('b t d, b i d -> b t i', *einsum_args)
                        logits = model.temperature.exp() * image_features @ classifier                             
                    else:
                        image_features = model.encode_image(images)
                        image_features = F.normalize(image_features, dim=-1)
                        logits = 100. * image_features @ classifier
            
            # measure accuracy with objectnet adjustments
            if split == "objectnet" and args.integer_labels:
                with open("./metadata/imagenet_to_objectnet.json","r") as f:
                    mapping = json.load(f)
                    # convert string keys to ints
                    mapping = {int(k): v for k, v in mapping.items()}
                pred = output.topk(max(topk), 1, True, True)[1].t()
                pred = torch.tensor(imageNetIDToObjectNetID[pred.cpu().tolist()]).to(args.device)
                #deal with the -1 wrong predictions, if need be
                correct = pred.eq(target.view(1, -1).expand_as(pred))
                acc1, acc5 = [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]          
            else:
                args.logits.append(logits.cpu().detach().numpy())
                if args.extended_metrics:
                    log_confusion_matrix(args, logits, target)
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    #TODO: debug integer labels for extended metrics
    if args.extended_metrics:
        write_confusion_matrix(args, logits, target, args.classnames)
    return top1, top5

def to_upper(l):
    return [c.upper() for c in l]

def to_lower(l):
    return [c.lower() for c in l]

def build_imagenet(args, model, in_type=""):
    isint = (args.integer_labels or args.linear_probe)
    usecaps = args.caption_subset and not isint
    if isint:
        args.classnames = get_imagenet_classnames()
        classifier = None
        return classifier
    template = get_openai_imagenet_template()
    if args.no_ensembling:
        template = [template[0]]
    if in_type == "r":
        if args.ds_cipher:
            classnames = get_imagenet_r_cipher()
        elif usecaps:
            classnames = get_imagenet_common_ir_classnames()
        else:
            classnames = get_imagenet_r_classnames()
    elif in_type == "a":
        if args.ds_cipher:
            classnames = get_imagenet_a_cipher()
        elif usecaps:
            classnames = get_imagenet_common_ia_classnames()
        else:
            classnames = get_imagenet_a_classnames()
    else:
        if args.ds_cipher:
            classnames = get_imagenet_cipher()
        elif usecaps:
            classnames = get_imagenet_cap_classnames()
        else:
            classnames = get_imagenet_classnames()
    if args.zs_upper:
        classnames = to_upper(classnames)
    elif args.zs_lower:
        classnames = to_lower(classnames)
    elif args.shift_cipher:
        classnames = [shift_cipher(s, args.shift_cipher) for s in classnames]
    #logging.info("imagenet classnames first 15: {}".format(classnames[:15]))
    args.classnames = classnames
    if not isint:
        logging.info('Building zero-shot classifier')
        classifier = zero_shot_classifier(model, classnames, template, args)
    return classifier

def imageNetIDToObjectNetID(prediction_class):
    for i in range(len(prediction_class)):
        if prediction_class[i] in mapping:
            prediction_class[i] = mapping[prediction_class[i]]
        else:
            prediction_class[i] = -1

def zero_shot_eval(model, data, epoch, args):
    #logging.debug(data)
    
    results = {}
    classifier = None

    if 'imagenet-val' not in data and 'imagenet-v2' not in data and 'imagenet-r' not in data and 'imagenet-s' not in data and 'imagenet-a' not in data and 'inat2021' not in data and 'stanfordcars' not in data and 'flowers' not in data and 'food' not in data and 'objectnet' not in data and 'insecta' not in data:
        return results
    if args.zeroshot_frequency == 0:
        return results
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return results

    if 'inat2021' in data:
        isint = (args.integer_labels or args.linear_probe)
        # usecaps = args.caption_subset and not isint
        if isint:
            args.classnames = inat_classnames
            classifier = None
            # return classifier
        else:
            logging.info('Building zero-shot classifier')
            classifier = zero_shot_classifier(model, inat_classnames, inat_template, args)
        # classifier = None
            logging.info('Using classifier')
        logging.info('Using classifier')
        top1, top5 = run(model, classifier, data['inat2021'].dataloader, args)
        results['inat2021-top1'] = top1
        results['inat2021-top5'] = top5

        logging.info('Finished zero-shot inat2021. Top1 was {}, top5 was {}'.format(top1, top5))

    if 'stanfordcars' in data:
        # if args.zs_upper:
        #     cars_classnames = to_upper(cars_classnames)
        # elif args.zs_lower:
        #     cars_classnames = to_lower(cars_classnames)
        logging.info("Starting zero-shot stanfordcars.")
        logging.info('Building zero-shot classifier')
        classifier = zero_shot_classifier(model, cars_classnames, cars_template, args)

        logging.info('Using classifier')
        top1, top5 = run(model, classifier, data['stanfordcars'].dataloader, args)
        results['stanfordcars-top1'] = top1
        results['stanfordcars-top5'] = top5

        logging.info('Finished zero-shot stanfordcars. Top1 was {}, top5 was {}'.format(top1, top5))

    if 'air' in data:
        logging.info("Starting zero-shot FGVC-aircraft.")
        logging.info('Building zero-shot classifier')
        classifier = zero_shot_classifier(model, air_classnames, air_template, args)

        logging.info('Using classifier')
        top1, top5 = run(model, classifier, data['air'].dataloader, args)
        results['FGVC-aircraft-top1'] = top1
        results['FGVC-aircraft-top5'] = top5

        logging.info('Finished zero-shot FGVC-aircraft. Top1 was {}, top5 was {}'.format(top1, top5))

    if 'food' in data:
            logging.info("Starting zero-shot food.")
            logging.info('Building zero-shot classifier')
            classifier = zero_shot_classifier(model, food_classnames, food_template, args)

            logging.info('Using classifier')
            top1, top5 = run(model, classifier, data['food'].dataloader, args)
            results['food-top1'] = top1
            results['food-top5'] = top5

            logging.info('Finished zero-shot food. Top1 was {}, top5 was {}'.format(top1, top5))

    if 'insecta' in data:

            isint = (args.integer_labels or args.linear_probe)
            # usecaps = args.caption_subset and not isint
            if isint:
                args.classnames = get_insecta_classnames()
                classifier = None
                # return classifier
            else:
                logging.info("Starting zero-shot insecta.")
                logging.info('Building zero-shot classifier')
                classifier = zero_shot_classifier(model, get_insecta_classnames(), inat_template, args)

            logging.info('Using classifier')
            top1, top5 = run(model, classifier, data['insecta'].dataloader, args)
            results['insecta-top1'] = top1
            results['insecta-top5'] = top5

            logging.info('Finished zero-shot insecta. Top1 was {}, top5 was {}'.format(top1, top5))

    logging.info('Starting zero-shot imagenet.')
    if args.caption_subset:
        logging.info("Using caption subset")
    isint = args.linear_probe or args.integer_labels
    classifier = None
    imagenets = []
    if 'imagenet-val' in data:            
        if classifier is None:
            classifier = build_imagenet(args, model)
        top1, top5 = run(model, classifier, data['imagenet-val'].dataloader, args, get_icap_idx() if args.caption_subset else None)
        results['imagenet-zeroshot-val-top1'] = top1
        imagenets.append(top1)
        results['imagenet-zeroshot-val-top5'] = top5
        logging.info('Finished zero-shot val. Top1 was {}, top5 was {}'.format(top1, top5))
    if 'imagenet-v2' in data:
        if classifier is None:
            classifier = build_imagenet(args, model)
        top1, top5 = run(model, classifier, data['imagenet-v2'].dataloader, args, get_icap_idx() if args.caption_subset else None)
        results['imagenetv2-zeroshot-val-top1'] = top1
        imagenets.append(top1)
        results['imagenetv2-zeroshot-val-top5'] = top5
        logging.info('Finished zero-shot v2. Top1 was {}, top5 was {}'.format(top1, top5))
    if 'imagenet-s' in data:
        if classifier is None:
            classifier = build_imagenet(args, model)
        top1, top5 = run(model, classifier, data['imagenet-s'].dataloader, args, get_icap_idx() if args.caption_subset else None)
        results['imagenets-zeroshot-val-top1'] = top1
        imagenets.append(top1)
        results['imagenets-zeroshot-val-top5'] = top5
        logging.info('Finished zero-shot sketch. Top1 was {}, top5 was {}'.format(top1, top5))
    if 'imagenet-r' in data:
        classifier = build_imagenet(args, model, "r")
        if isint:
            top1, top5 = run(model, classifier, data['imagenet-r'].dataloader, args, get_common_ir_idx() if args.caption_subset else get_ir_idx(), "r")
        else:
            top1, top5 = run(model, classifier, data['imagenet-r'].dataloader, args, get_common_ir_idx_zeroindexed() if args.caption_subset else None, "r")
        results['imagenetr-zeroshot-val-top1'] = top1
        imagenets.append(top1)
        results['imagenetr-zeroshot-val-top5'] = top5
        logging.info('Finished zero-shot imagenet-r. Top1 was {}, top5 was {}'.format(top1, top5))
    if 'imagenet-a' in data:
        classifier = build_imagenet(args, model, "a")
        if isint:
            top1, top5 = run(model, classifier, data['imagenet-a'].dataloader, args, get_common_ia_idx() if args.caption_subset else get_ia_idx(), "a")
        else:
            top1, top5 = run(model, classifier, data['imagenet-a'].dataloader, args, get_common_ia_idx_zeroindexed() if args.caption_subset else None, "a")
        results['imageneta-zeroshot-val-top1'] = top1
        imagenets.append(top1)
        results['imageneta-zeroshot-val-top5'] = top5
        logging.info('Finished zero-shot imagenet-a. Top1 was {}, top5 was {}'.format(top1, top5))
    if 'objectnet' in data:
        obj_classnames = ast.literal_eval(open("./metadata/objectnet_folder_to_label.txt", 'r').read())
        obj_classnames = sorted(obj_classnames.values())
        classifier = zero_shot_classifier(model, obj_classnames, openai_imagenet_template, args)
        top1, top5 = run(model, classifier, data['objectnet'].dataloader, args, None, "objectnet")
        results['objectnet-top1'] = top1
        results['objectnet-top5'] = top5        
    if results.get('imagenet-zeroshot-val-top1'):
        logging.info("computing effective robustness on imagenet")
        logging.info("len imagenets {}".format(len(imagenets)))
        try:
            imagenet_shifts = []
            for shift in ['imagenetr-zeroshot-val-top1', 'imageneta-zeroshot-val-top1', 'imagenets-zeroshot-val-top1', 'imagenetv2-zeroshot-val-top1']:
                if results.get(shift):
                    imagenet_shifts.append(results[shift])
            if len(imagenet_shifts) > 0:
                results['imagenet-average-robustness'] = np.average(imagenet_shifts)
                results['imagenet-effective-robustness'] = np.divide(np.average(imagenet_shifts), results['imagenet-zeroshot-val-top1'])
                logging.info("Average robustness over {} ImageNet shifts: {}".format(len(imagenet_shifts), results['imagenet-average-robustness']))
        except Exception as e:
            logging.info("error calculating effective robustness: ")
            logging.info(e)
    logging.info('Finished zero-shot evals')

    return results