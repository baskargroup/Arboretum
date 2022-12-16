import torch
import torch.nn as nn
from torch.nn import functional as F
import logging
from torch.autograd import Variable
import torchqmet

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from .utils import all_gather_batch_with_grad, GatherLayer

def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)
    return all_image_features, all_text_features

class ClipLoss(nn.Module):

    def __init__(
            self,
            img_weight=.5,
            text_weight=.5,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.img_weight = img_weight
        self.text_weight = text_weight
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            #FIXME: band-aid handling of nans
            if torch.any(torch.isnan(all_image_features)):
                logging.warning("found NaN in images, replacing with a small number")
                all_image_features = torch.nan_to_num(all_image_features, nan=1e-10)
            if torch.any(torch.isnan(all_text_features)):
                logging.warning("found NaN in texts, replacing with a small number")
                all_text_features = torch.nan_to_num(all_text_features, nan=1e-10)
            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            #FIXME: band-aid handling of nans
            if torch.any(torch.isnan(image_features)):
                logging.warning("found NaN in images, replacing with a small number")
                image_features = torch.nan_to_num(image_features, nan=1e-10)
            if torch.any(torch.isnan(text_features)):
                logging.warning("found NaN in texts, replacing with a small number")
                text_features = torch.nan_to_num(text_features, nan=1e-10)
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        total_loss = F.cross_entropy(logits_per_image, labels) * self.img_weight + F.cross_entropy(logits_per_text, labels) * self.text_weight
        if torch.any(torch.isnan(total_loss)):
            logging.warning("Leaving ClipLoss, NaN loss detected: {}".format(total_loss))
        return total_loss

class ClipLossIQE(nn.Module):

    def __init__(
            self,
            img_weight=.5,
            text_weight=.5,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.img_weight = img_weight
        self.text_weight = text_weight
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        d_iqe = torchqmet.IQE(
                input_size=image_features.size(dim=1),
                dim_per_component=16,      # split dimensions into 16-dimensional chunks, where each chunk
                reduction="sum"              #    gives an IQE component (IQE paper recommends `dim_per_component >= 8`)
            ).to(device)
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            #FIXME: band-aid handling of nans
            if torch.any(torch.isnan(all_image_features)):
                logging.warning("found NaN in images, replacing with a small number")
                all_image_features = torch.nan_to_num(all_image_features, nan=1e-10)
            if torch.any(torch.isnan(all_text_features)):
                logging.warning("found NaN in texts, replacing with a small number")
                all_text_features = torch.nan_to_num(all_text_features, nan=1e-10)
            logits = logit_scale * torch.outer(d_iqe(all_image_features, all_text_features) * self.img_weight, d_iqe(all_text_features, all_image_features) * self.text_weight)
            # logits = logit_scale * (d_iqe(all_image_features[:, None], all_text_features) * self.img_weight + d_iqe(all_text_features[:, None], all_image_features) * self.text_weight)
        else:
            #FIXME: band-aid handling of nans
            if torch.any(torch.isnan(image_features)):
                logging.warning("found NaN in images, replacing with a small number")
                image_features = torch.nan_to_num(image_features, nan=1e-10)
            if torch.any(torch.isnan(text_features)):
                logging.warning("found NaN in texts, replacing with a small number")
                text_features = torch.nan_to_num(text_features, nan=1e-10)
            logits = logit_scale * torch.outer(d_iqe(image_features, text_features) * self.img_weight,  d_iqe(text_features, image_features) * self.text_weight) 
            # logits = logit_scale * (d_iqe(image_features[:, None], text_features) * self.img_weight + d_iqe(text_features[:, None], image_features) * self.text_weight)
        # logging.debug("logits")
        # logging.debug(logits)
        # logging.debug("shape")
        # logging.debug(str(logits.size()))
        # calculated ground-truth and cache if enabled
        num_logits = logits.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        total_loss = F.cross_entropy(logits, labels)
        if torch.any(torch.isnan(total_loss)):
            logging.warning("Leaving ClipLossIQE, NaN loss detected: {}".format(total_loss))
        return total_loss

class SIMCLRLoss(nn.Module):
    """
    This is the SimCLR loss in https://arxiv.org/abs/2002.05709
    As implemented in https://github.com/facebookresearch/SLIP/blob/main/losses.py
    The embedding vectors are assumed to have size (2 x batch_size, embedding_dim) and
    the memory layout that can be reshaped into shape (2, batch_size, embedding_dim).
    This memory layout is consistent with the SimCLR collator in
    https://github.com/facebookresearch/vissl/blob/master/vissl/data/collators/simclr_collator.py
    Config params:
        temperature (float): the temperature to be applied on the logits
    """

    def __init__(self, temperature=0.1, args=None):
        super().__init__()
        self.tau = temperature
        self.labels = None
        self.masks = None
        self.last_local_batch_size = None
        self.args = args

    def forward(self, outputs):
        q_a = outputs['aug1_embed']
        q_b = outputs['aug2_embed']

        q_a = F.normalize(q_a, dim=-1, p=2)
        q_b = F.normalize(q_b, dim=-1, p=2)

        local_batch_size = q_a.size(0)
        if self.args.world_size > 1:
            k_a, k_b = gather_features(
                q_a, q_b,
                self.args.local_loss, self.args.gather_with_grad, self.args.rank, self.args.world_size)
        else:
            k_a = q_a
            k_b = q_b
        # k_a, k_b = all_gather_batch_with_grad([q_a, q_b], self.world_size)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * self.args.rank + torch.arange(
                local_batch_size, device=q_a.device
            )
            total_batch_size = local_batch_size * self.args.world_size
            self.masks = F.one_hot(self.labels, total_batch_size) * 1e9
            self.last_local_batch_size = local_batch_size

        logits_aa = torch.matmul(q_a, k_a.transpose(0, 1)) / self.tau
        logits_aa = logits_aa - self.masks
        logits_bb = torch.matmul(q_b, k_b.transpose(0, 1)) / self.tau
        logits_bb = logits_bb - self.masks
        logits_ab = torch.matmul(q_a, k_b.transpose(0, 1)) / self.tau
        logits_ba = torch.matmul(q_b, k_a.transpose(0, 1)) / self.tau

        loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), self.labels)
        loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), self.labels)
        loss = (loss_a + loss_b) / 2  # divide by 2 to average over all samples

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(torch.cat([logits_ab, logits_aa], dim=1), dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / local_batch_size

        return {'loss': loss, 'ssl_loss': loss, 'ssl_acc': acc}
    
class IntLoss(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.last_local_batch_size = None

    def forward(self, logits, labels):
        if self.args.world_size > 1:
            local_batch_size = logits.size(0)
            logits, labels = gather_features(
                logits, labels,
                self.args.local_loss, self.args.gather_with_grad, self.args.rank, self.args.world_size)
            local_range = local_batch_size * self.args.rank + torch.arange(local_batch_size, device=self.device)
            logits = logits[local_range]
            labels = labels[local_range]
        loss = 0
        lablen = 0
        try:
            for l in labels:
                # logging.info(l)
                if len(l) > lablen:
                    lablen = len(l)
        except Exception as e:
            #logging.warning(e)
            pass
        if lablen > 1:
            #logging.info("Entering multiclass, length {}".format(lablen))
            for idx, label in enumerate(labels):
                for tag in label:
                    if tag == -1:
                        break
                    pred = logits[idx]
                    loss = loss + F.cross_entropy(pred, tag)
        else:
            loss = F.cross_entropy(logits, labels)
        return loss

    #Other interesting loss functions
    #https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py

class ClipLossAlignUnif(nn.Module):

    def __init__(
            self,
            img_weight=.5,
            text_weight=.5,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.img_weight = img_weight
        self.text_weight = text_weight
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def align_loss(self, x, y, alpha=2):
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    def uniform_loss(self, x, t=2):
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            #FIXME: band-aid handling of nans
            if torch.any(torch.isnan(all_image_features)):
                logging.warning("found NaN in images, replacing with a small number")
                all_image_features = torch.nan_to_num(all_image_features, nan=1e-10)
            if torch.any(torch.isnan(all_text_features)):
                logging.warning("found NaN in texts, replacing with a small number")
                all_text_features = torch.nan_to_num(all_text_features, nan=1e-10)
            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            #FIXME: band-aid handling of nans
            if torch.any(torch.isnan(image_features)):
                logging.warning("found NaN in images, replacing with a small number")
                image_features = torch.nan_to_num(image_features, nan=1e-10)
            if torch.any(torch.isnan(text_features)):
                logging.warning("found NaN in texts, replacing with a small number")
                text_features = torch.nan_to_num(text_features, nan=1e-10)
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.diag(torch.ones(num_logits)).to(device)
            # if self.world_size > 1 and self.local_loss:
            #     labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        #print("labels")
        #print(labels)
        #print("input sizes")
        m = nn.Softmax(dim=1)
        im_preds = m(logits_per_image)
        txt_preds = m(logits_per_text)
        #print(im_preds.size(), labels.size())
        align_loss = self.align_loss(logits_per_image, logits_per_text)
        unif_loss_img = self.uniform_loss(logits_per_image)
        unif_loss_txt = self.uniform_loss(logits_per_text)
        #print("losses")
        #print(align_loss.size(), unif_loss_img.size(), unif_loss_txt.size())
        total_loss = align_loss + unif_loss_img / 2 + unif_loss_txt / 2
        if torch.any(torch.isnan(total_loss)):
            logging.warning("NaN detected leaving loss function: {}".format(total_loss))
        return total_loss