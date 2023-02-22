import json
import torch
from pathlib import Path
import PIL


METADATA = Path("./metadata")

with open(METADATA / 'folder_to_objectnet_label.json', 'r') as f:
	folder_map = json.load(f)
	folder_map = {v: k for k, v in folder_map.items()}

with open(METADATA / 'objectnet_to_imagenet_1k.json', 'r') as f:
	objectnet_map = json.load(f)

with open(METADATA / 'imagenet_to_labels.json', 'r') as f:
	imagenet_map = json.load(f)
	imagenet_map = {v: k for k, v in imagenet_map.items()}


folder_to_ids, class_sublist = {}, []
for objectnet_name, imagenet_names in objectnet_map.items():
    imagenet_names = imagenet_names.split('; ')
    imagenet_ids = [int(imagenet_map[imagenet_name]) for imagenet_name in imagenet_names]
    class_sublist.extend(imagenet_ids)
    folder_to_ids[folder_map[objectnet_name]] = imagenet_ids

# def crop_image(image, border=2):
# 	return PIL.ImageOps.crop(image, border=border)

def objectnet_accuracy(logits, targets, image_paths, using_class_sublist=True, in100=False):
    if in100:
        folder_map = {'banana': [0], 'beer_bottle': [1], 'bottle_cap' : [2], 'lampshade' : [3], 'laptop_open' : [4], 'lemon' : [5], 'necklace' : [6], 'orange' : [7], 'padlock' : [8], 'pillow' : [9], 'teapot' : [10], 'umbrella' : [11], 'wine_bottle' : [12], 'winter_glove' : [13]}
    elif using_class_sublist:
        folder_map = {k: [class_sublist.index(x) for x in v] for k, v in folder_to_ids.items()}
    else:
        folder_map = folder_to_ids
    preds = logits.argmax(dim=1)
    num_correct, num_total = 0, 0
    for pred, image_path in zip(preds, image_paths):
        folder = Path(image_path).parent.name
        if folder in folder_map:
            num_total += 1
            if pred in folder_map[folder]:
                num_correct += 1
    if num_total == 0:
        return None
    return (num_correct, num_total)

# def imageNetIDToObjectNetID(prediction_class):
#     for i in range(len(prediction_class)):
#         if prediction_class[i] in mapping:
#             prediction_class[i] = mapping[prediction_class[i]]
#         else:
#             prediction_class[i] = -1

# def objectnet_integer_accuracy(logits, targets, image_paths, using_class_sublist=True, in100=False):
#     with open("./metadata/imagenet_to_objectnet.json","r") as f:
#         mapping = json.load(f)
#             # convert string keys to ints
#         mapping = {int(k): v for k, v in mapping.items()}
#         if in100:
#         else:
#             logits[]
#         pred = logits.topk(max(topk), 1, True, True)[1].t()
#         pred = torch.tensor(imageNetIDToObjectNetID[pred.cpu().tolist()]).to(args.device)
#         #deal with the -1 wrong predictions, if need be
#         correct = pred.eq(target.view(1, -1).expand_as(pred))
#         acc1, acc5 = [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]  

idx_subsample_list = [range(x*50, (x+1)*50) for x in class_sublist]
idx_subsample_list = sorted([item for sublist in idx_subsample_list for item in sublist])

folder_map_sublisted = {k: [class_sublist.index(x) for x in v] for k, v in folder_to_ids.items()}
objectnet_idx_to_imgnet_idxs = {idx: folder_map_sublisted[name] for idx, name in enumerate(sorted(folder_map_sublisted))}
imgnet_idx_to_objectnet_idx =  {}
for objectnet_idx, imgnet_idxs in objectnet_idx_to_imgnet_idxs.items():
	for imgnet_idx in imgnet_idxs:
		imgnet_idx_to_objectnet_idx[imgnet_idx] = objectnet_idx

def accuracy_topk(logits, targets, topk=(1, 5)):
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = {}
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res[f'top{k}'] = correct_k.mul_(100.0 / batch_size).item()
    return res

def accuracy_topk_subselected_and_collapsed(logits, targets):
    collapsed_logits = torch.zeros((logits.size(0), len(folder_map_sublisted)), dtype=logits.dtype, device=logits.device)
    for objectnet_idx, imgnet_idxs in objectnet_idx_to_imgnet_idxs.items():
        collapsed_logits[:, objectnet_idx] = logits[:, imgnet_idxs].max(dim=1).values
    target_list = []
    for x in targets:
        if class_sublist[x.item()] in imgnet_idx_to_objectnet_idx:
            target_list.append(imgnet_idx_to_objectnet_idx[class_sublist.index(x.item())])
    if len(target_list) == 0:
        return None
    return (accuracy_topk(collapsed_logits, torch.tensor(target_list)), len(target_list))