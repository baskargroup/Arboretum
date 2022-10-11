import ast
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from multiprocessing import Value
from functools import partial

import braceexpand
import numpy as np
import pandas as pd
from tqdm.auto import tqdm  # for notebooks
tqdm.pandas()
import torch
import torchvision
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample
from clean_filter_captions import *

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

try:
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('tagsets')
    from nltk.corpus import stopwords, wordnet
    lemmatizer = nltk.stem.WordNetLemmatizer()

except:
    logging.warning("nltk load failed, filtering not available")

from open_clip import tokenize

from .imagenet_zeroshot_data import *

try:
    from .inat_zeroshot_data import inat_classnames, inat_template
    from .cars_zeroshot_data import cars_classnames, cars_template
    from .flowers_zeroshot_data import flowers_classnames, flowers_template
    from .food_zeroshot_data import food_classnames, food_template
    from .air_zeroshot_data import air_classnames, air_template

except Exception as e:
    logging.info("Import exception: ")
    logging.info(e)

cipher_dict = {
    'a': '@^',
    'b': '#^',
    'c': '$^',
    'd': '%^',
    'e': '@&',
    'f': '#&',
    'g': '$&',
    'h': '%&',
    'i': '@*',
    'j': '#*',
    'k': '$*',
    'l': '%*',
    'm': '@(',
    'n': '#(',
    'o': '$(',
    'p': '%(',
    'q': '@)',
    'r': '#)',
    's': '$)',
    't': '%)',
    'u': '@+',
    'v': '#+',
    'w': '$+',
    'x': '%+',
    'y': '@=',
    'z': '#=',
    ' ': ' ',
    '\'': '\'',
    ',': ',',
    '.': '.',
    '!': '!',
    '?': '?',
}

total_count = [0]

def get_total_obj():
    return globals()['total_count']

class TotalSize:
    """Keep track of the total size of samples."""

    def __init__(self):
        """Create a TotalSize counter."""
        self.count = 0

    def __call__(self, sample):
        """Add sample to the counter.
        :param sample: undecoded sample to be added
        """
        self.count += 1
        total_count[0] += 1
        if total_count[0] % 1000 == 0:
            logging.debug("Total samples seen is now {} (times number of workers)".format(total_count[0]))
        return sample

def select_count(data, predicate, count):
    """Select samples based on a predicate.
    :param data: source iterator
    :param predicate: predicate (function)
    """
    for sample in data:
        if predicate(sample):
            count = count + 1
            yield sample

def token_strip_func(texts):
    ok_tokens = set(ast.literal_eval(open("/scratch/bf996/open_clip/metadata/in1k_ds_oai_tokens.txt", 'r').read()))
    tlist = texts.tolist()
    texts = [t if t in ok_tokens else 0 for t in tlist]
    texts = torch.tensor(tlist)
    return texts

def clean_integer_label(label, singleclass, strict):
    if isinstance(label, float):
        label = int(label)
    if isinstance(label, int):
        if label < 0 or label > 999:
            logging.info("Integer label {} out of acceptable range, mapping to 0".format(label))
            label = 0
        if singleclass:
            return torch.tensor(label)
        else:
            label_l = [label] + [-1 for i in range(24)]
            return torch.tensor(label_l)
    elif isinstance(label, str) and label == "":
        return ""
    elif isinstance(label, str):
        label = label.split(", ")
        try:
            label_updated = []
            for l in label:
                intl = int(l)
                if intl < 0 or intl > 999:
                    logging.info("Integer label {} out of acceptable range, mapping to 0".format(intl))
                    label_updated.append(0)
                else:
                    label_updated.append(intl)
            label = label_updated
        except Exception as e:
            logging.warning("Error converting string to integer list, {}".format(e))
            return ""
        if label == []:
            logging.warning("No integers found in {}. Returning false.".format(label))
            return ""
        if 1 <= len(label) < 25:
            padding = [-1 for i in range(25 - len(label))]
            label = label + padding
        elif len(label) > 25:
            label = label[:25]
        if len(label) != 25:
            logging.warning("Integer label {} has length {}".format(label, len(label)))
        if singleclass:
            label = int(label[0])
        elif strict:
            if label[1] == -1:
                label = int(label[0])
            else:
                return ""
        return torch.tensor(label)
    else:
        logging.warning("Expected string or int or float, got {} -- ignoring".format(type(label)))
        return ""

class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, csvfilter, csvscrambled, tokenscrambled, csvcleaned, dscipher, simplecaptions, strict, shift, integer_labels, multiclass, metacaptions, token_strip, sep="\t"):
        logging.debug(f'Loading csv data from {input_filename}')
        df = pd.read_csv(input_filename, sep=sep)
        df = df[df[caption_key].notnull()]
        df = df[df[caption_key] != "nan"]
        if integer_labels:
            df = df[df[caption_key] != -1]
            df = df[df[caption_key] != "-1"]
        logging.debug("Columns of dataframe: {}".format(df.columns))
        if dscipher:
            csvcleaned=True
        if csvcleaned:
            logging.debug('Cleaning captions. Original dataset size is {}'.format(len(df)))
            df.loc[:, caption_key] = df.title.progress_apply(clean_captions)
            df = df[df['title'].str.len() > 0]
            logging.debug("Done. Length is now {}".format(len(df)))
            logging.debug(df.head())
        if dscipher or simplecaptions or shift:
            logging.debug('Transforming or encoding captions. Original dataset size is {}'.format(len(df)))
            df['title'] = df[caption_key].progress_apply(synset_ds, ngram=3, ds=csvfilter, cipher=dscipher, simplecaptions=simplecaptions, strict=strict, shift=shift, metacaptions=metacaptions)
            logging.debug(df['title'].head())
            df = df[df['title'].str.len() > 0]
            logging.debug("Done. Length is now {}".format(len(df)))
            logging.debug(df.head())            
        elif csvfilter != "":
            logging.debug('Filtering captions. Original dataset size is {}'.format(len(df)))
            df['is_synset'] = df[caption_key].progress_apply(synset_ds, ngram=3, ds=csvfilter, cipher=False, simplecaptions=False, strict=strict, shift=shift, metacaptions=metacaptions)
            logging.debug(df['is_synset'].head())
            df = df[df['is_synset']].drop(columns=['is_synset'])
            logging.debug("Done. Length is now {}".format(len(df)))
            logging.debug(df.head())
        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        self.scrambled = csvscrambled
        self.csvfilter = csvfilter
        self.token_scrambled = tokenscrambled
        self.integer_labels = integer_labels
        self.multiclass = multiclass
        self.strict = strict
        self.token_strip = token_strip
        logging.debug('Done loading data')

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        try:
            images = self.transforms(Image.open(str(self.images[idx])))
            # assert(images.size() == (3, 224, 224))
            torch.nan_to_num(images, nan=0.01, posinf=0.99, neginf=0.01)
            texts = str(self.captions[idx])
        except Exception as e:
            logging.warning("Exception in csv dataset: {}".format(e))
            logging.warning("Missing or unreadable image at {}, attempting to skip.".format(str(self.images[idx])))
            try:
                images = self.transforms(Image.open(str(self.images[idx+1])))
                torch.nan_to_num(images, nan=0.01, posinf=0.99, neginf=0.01)
                texts = str(self.captions[idx+1])
            except:
                logging.warning("Skip failed. Generating dummy image and caption.".format(str(self.images[idx])))
                imarray = np.random.rand(224,224,3) * 255
                images = self.transforms(
                    Image.fromarray(imarray.astype('uint8')).convert('RGBA')
                    )
                texts = "NONE"
        if self.integer_labels:
            #if isinstance(texts, str) and not texts.is_numeric():
                #assert(False, "Integer labels cannot be computed on the fly for a CSV dataset")
                #texts = [synset_ds(clean_captions(str(texts)), 3, self.csvfilter, False, False, self.strict, False, True, None) for t in texts]
            texts = clean_integer_label(self.captions[idx], not self.multiclass, self.strict)
            return images, texts
        if self.scrambled:
            texts = scramble_txt(texts)
        texts = tokenize(texts)[0]
        if self.token_strip:
            texts = token_strip_func(texts)
        if self.token_scrambled:
            #logging.info("before: {}".format(texts))
            random.shuffle(texts)
            #logging.info("after: {}".format(texts))
        return images, texts

def _convert_to_rgb(image):
    return image.convert('RGB')
    
class ImageAugCSVDataset(CsvDataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, csvfilter, csvscrambled, csvcleaned, dscipher, simplecaptions, strict, shift, integer_labels, metacaptions, token_strip, sep="\t"):
        super().__init__(input_filename, transforms, img_key, caption_key, csvfilter, csvscrambled, csvcleaned, dscipher, simplecaptions, strict, shift, integer_labels, metacaptions, token_strip, sep)
        self.augment = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            torchvision.transforms.RandomApply([
                torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
            torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur(5, sigma=(.1, 2.))], p=0.5),
            torchvision.transforms.RandomHorizontalFlip(),
            _convert_to_rgb,
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean = (0.48145466, 0.4578275, 0.40821073), std = (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        try:
            img = Image.open(str(self.images[idx]))
        except Exception as e:
            logging.warning("Exception in csv dataset: {}".format(e))
            logging.warning("Missing or unreadable image at {}, attempting to skip.".format(str(self.images[idx])))
            try:
                img = Image.open(str(self.images[idx+1]))
            except:
                logging.warning("Skip failed. Generating dummy image and caption.".format(str(self.images[idx])))
                imarray = np.random.rand(224,224,3) * 255
                img = Image.fromarray(imarray.astype('uint8')).convert('RGB')
        try:
            aug1 = self.augment(img)
            aug2 = self.augment(img)
        except Exception as e:
            logging.info("Exception during augmentation: {} \n Generating dummy image and caption".format(e))
            imarray = np.random.rand(224,224,3) * 255
            img = Image.fromarray(imarray.astype('uint8')).convert('RGB')
            aug1 = self.augment(img)
            aug2 = self.augment(img)
        return aug1, aug2

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def preprocess_txt(text, token_scrambled, token_strip):
    text = str(text)
    tokentxt = tokenize([text])[0]
    if token_scrambled:
        random.shuffle(tokentxt)
    if token_strip:
        text = token_strip_func(text)
    return tokentxt

def filter_preprocess_txt(text, ds, scrambled, dscipher, simplecaptions, strict, shift, integer_labels, multiclass, metacaptions):
    if bool(ds):
        if integer_labels:
            text = clean_captions(str(text))
            text = synset_ds(text, 3, ds, False, False, strict, False, integer_labels, metacaptions)
            if text:
                text = clean_integer_label(text, not multiclass, strict)
            else:
                text = ""
        else:
            text = clean_captions(str(text))
            if not synset_ds(text, 3, ds, False, False, strict, shift, integer_labels, metacaptions):
                text = ""
    elif any([dscipher, simplecaptions, strict, shift]):
        text = clean_captions(str(text))
        text = synset_ds(text, 3, ds, dscipher, simplecaptions, strict, shift, integer_labels, metacaptions)
        if not text:
            text = ""
    if scrambled:
        text = scramble_txt(text)
    return text

def scramble_txt(text):
    tlist = text.split(" ")
    random.shuffle(tlist)
    text = " ".join(tlist).strip()   
    return text 

"""
Shift cipher for alphabetic strings
"""

def shift_cipher(s, shift):
  retstr = ""
  for c in s:
    if c.isalpha():
      if c.islower():
          ordshift = 97
      else:
          ordshift = 65
      c = c.translate({ord(ch):(ord(ch) - ordshift + shift) % 26 + ordshift for ch in c})
    retstr = retstr + c
  return retstr

"""
Synset builder

Dataset argument expects a list of class names as strings -- any dataset can be used
Strict follows the methodology of Fang et al ... multiple matches -> no match
nva uses parts of speech for all of wordnet, instead of matching on some list from a dataset
WARNING: can return string or bool, depending on arguments provided
"""

def synset_ds(s, ngram=3, ds=None, cipher=False, simplecaptions=False, strict=False, shift=None, integer_labels=False, metacaptions=None):
    flag = False
    s = list(lemmatizer.lemmatize(t) for t in s.split(" "))
    str_s = " ".join(w for w in s)
    if ds:
        ds_values = ds_val_getter(ds, False)
    for count, word in enumerate(s):
        if strict and flag:
            break
        grams = []
        for i in range(ngram):
            if count + i - 1 > len(s):
                continue
            gram = " ".join(w for w in s[count:count+i+1])
            grams.append(gram)
        for i, gram in enumerate(grams):
            if strict and flag:
                break
            gram_t = gram
            if cipher:
                k = ""
                for c in gram:
                    nextc = cipher_dict.get(c) or c
                    k = k + nextc
                gram_t = k
            if ds:
                for idx, val in enumerate(ds_values.values()):
                    if gram_t in val:
                        if integer_labels and not flag:
                            str_s = "{}".format(idx)
                        elif integer_labels:
                            idx_insert = idx
                            if str_s.find(str(idx_insert)) == -1:
                                str_s += ", {}".format(idx_insert)
                            continue
                        elif not metacaptions.empty:
                            idx_insert = idx
                            row = metacaptions[metacaptions['idx'].str.contains(str(idx_insert))]
                            if not row.empty:
                                row = row.iloc[0]
                                for field in [row['functional_classname'], row['subclass'], row['relations_locations'], row['relations_objects'], row['relations_events']]:
                                    if field == "":
                                        continue
                                    items = field.split(", ")
                                    for item in items:
                                        if str_s.find(item) == -1:
                                            str_s = str_s + " " + item
                        if simplecaptions and not flag:
                            str_s = "An image of " + gram_t
                        elif simplecaptions and flag and str_s.find(gram) == -1:
                            str_s += " {}".format(gram)
                        flag = True
                        if cipher:
                            str_s = str_s.replace(gram, k)

            elif simplecaptions and not ds:
                d = wordnet.synsets(gram)
                if d in stopwords.words('english'):
                    continue
                elif d and not flag:
                    str_s = "An image of " + gram
                elif d and str_s.find(gram) == -1:
                    str_s += " {}".format(gram)         
                flag=True
    
    if len(str_s) > 76:
        str_s = str_s[:75]
    
    if cipher or simplecaptions or integer_labels:
        if not flag:
            str_s = ""
        return str_s

    elif shift:
        str_s = shift_cipher(str_s, shift)
        return str_s

    return flag

def get_dataset_size(shards):
    shards_list = list(braceexpand.braceexpand(shards))
    dir_path = os.path.dirname(shards)
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards

def get_objectnet(args, preprocess_fns):
    preprocess_train, preprocess_val = preprocess_fns
    dataset = datasets.ImageFolder(args.objectnet, transform=preprocess_val)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            sampler=None
        )
    return DataInfo(dataloader=dataloader, sampler=None)

def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2", "r", "a", "s"], "Not a recognized ImageNet split, {}".format(split)
    is_train = (split == "train")
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    elif is_train:
        data_path = args.imagenet_train
        preprocess_fn = preprocess_train
        dataset = datasets.ImageFolder(data_path, transform=preprocess_train)
    else:
        if split == "val":
            data_path = args.imagenet_val
        if split == "r":
            data_path = args.imagenet_r
        if split == "a":
            data_path = args.imagenet_a
        if split == "s":
            data_path = args.imagenet_s
        preprocess_fn = preprocess_val
        assert data_path, "No data path found"

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)
    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)

def get_torchvision(args, preprocess_fns, ds):
    _, preprocess_val = preprocess_fns
    preprocess_fn = preprocess_val
    if ds == "stanfordcars":
        data_path = args.stanfordcars
        dataset = datasets.StanfordCars(root = data_path, split = 'test', transform = preprocess_fn, download = False)
    elif ds == "flowers":
        data_path = args.flowers
        dataset = datasets.Flowers102(root = data_path, split = 'test', transform = preprocess_fn, download = False)
    elif ds == "air":
        data_path = args.air
        dataset = datasets.FGVCAircraft(root=data_path, split = 'val', annotation_level = 'family', transform = preprocess_fn, download = False)
    elif ds == "food":
        data_path = args.food
        dataset = datasets.Food101(root = data_path, split = 'test', transform = preprocess_fn, download = False)
    elif ds == "inat2021":
        data_path = args.inat2021
        dataset = datasets.INaturalist(root = data_path, version = "2021_valid", transform = preprocess_fn, download = False)
    elif ds == "inat2018":
        data_path = args.inat2018
        dataset = datasets.INaturalist(root = data_path, version = "2018", transform = preprocess_fn, download = False)
    elif ds == "inat2017":
        data_path = args.inat2017
        dataset = datasets.INaturalist(root = data_path, version = "2017", transform = preprocess_fn, download = False)
    sampler = None
    dataloader = torch.utils.data.DataLoader(
    dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler
    )

    return DataInfo(dataloader, sampler)

def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts), "Length of images, texts did not match"
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    return ('txt' in sample) and ('png' in sample or 'jpg' in sample)

def filter_no_caption_text(sample):
    if 'text' not in sample:
        logging.warning("Text not in sample")
        return False
    elif sample["text"] == "":
        return False
    else:
        return True

def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict), "Filesample is a {}, not a dict".format(type(filesample))
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed():
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour the seed already created for pytorch dataloader workers if it exists
        return worker_info.seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            seed = pytorch_worker_seed() + epoch
        else:
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls = wds.shardlists.expand_urls(urls)
        self.urls = urls
        assert isinstance(self.urls[0], str), "self.urls[0] is a {}, not a string".format(type(self.urls[0]))
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = pytorch_worker_seed if worker_seed is None else worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic, worker seed should be deterministic due to arg.seed
            self.rng.seed(self.worker_seed() + epoch)
        for _ in range(self.nshards):
            yield dict(url=self.rng.choice(self.urls))

def replace_with_image(d):
    d['text'] = d['image']
    return d

def my_collate(batch):
    # logging.debug("batch contents: {}".format(batch))
    len_batch = len(batch) # original batch length
    # logging.debug("Before filter, batch length is {}".format(len_batch))
    batch = list(filter (lambda x:bool(x)==True, batch[0])) # filter out all the Nones
    batch = list(filter (lambda x:bool(x)==True, batch[1])) # filter out all the Nones
    # logging.debug("After filter, batch length is {}".format(len(batch)))
    if len_batch > len(batch): # if there are samples missing just use existing members, doesn't work if you reject every sample in a batch
        logging.info("Found empty samples in batch")
        diff = len_batch - len(batch)
        for i in range(diff):
            batch = batch + batch[:diff]
    return torch.utils.data.dataloader.default_collate(batch)

def intlabel_collate(batch):
    #batch is a tuple with BATCH_LEN items
    #batch[0] is a tuple with 2 items (image, label)
    c_batch = []
    for b in batch:
        if not torch.is_tensor(b):
            logging.warning("not a tensor, {}, skipping".format(b[1]))
        else:
            batch.append(b)
    return torch.utils.data.dataloader.default_collate(c_batch)

def get_wds_dataset_simclr(args, is_train, epoch=0, floor=False, total=None, num_samples=0, shared_epoch=0, pipeline=None):
    preprocess_img = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            torchvision.transforms.RandomApply([
                torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
            torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur(5, sigma=(.1, 2.))], p=0.5),
            torchvision.transforms.RandomHorizontalFlip(),
            _convert_to_rgb,
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean = (0.48145466, 0.4578275, 0.40821073), std = (0.26862954, 0.26130258, 0.27577711)),
    ])
    pipeline.extend([
        wds.map(replace_with_image),
        wds.map_dict(image=preprocess_img, text=preprocess_img, handler=log_and_continue),
    ])
    pipeline.extend([
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train),
    ])
    dataset = wds.DataPipeline(*pipeline)
    num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)

def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, total=None):
    if args.schema:
        input_shards = []
        num_samples = 0
        for k, schema in args.schemas.items():
            input_shards = input_shards + list(braceexpand.braceexpand(schema["train_data"]))
            num_samples += schema['train_num_samples']
        num_shards = len(input_shards)
    else:
        input_shards = args.train_data if is_train else args.val_data
        num_samples, num_shards = get_dataset_size(input_shards)
    assert input_shards is not None, "No input shards detected"
    resampled = getattr(args, 'dataset_resampled', False) and is_train
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError(
                    'Currently, number of dataset samples must be specified for training dataset. '
                    'Please specify via `--train-num-samples` if no dataset length info present.')
        else:
            num_samples = args.val_num_samples or 0  # eval will just exhaust the iterator if not specified

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    if resampled:
        pipeline = [ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)]
    else:
        shl = wds.SimpleShardList(input_shards)
        pipeline = [shl]

    # at this point we have an iterator over all the shards
    logging.debug("get_wds_dataset, is_train")
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png", text="txt"),
    ])
    if args.sim_clr:
        return get_wds_dataset_simclr(args, is_train, epoch=epoch, floor=floor, total=total, num_samples=num_samples, shared_epoch=shared_epoch, pipeline=pipeline)
    # if total:
    #     pipeline.extend([
    #         wds.map_dict(image=total, handler=log_and_continue)
    #     ])
    if any([args.ds_filter, args.csv_scrambled, args.ds_cipher, args.simplecaptions, args.strict, args.shift_cipher, args.integer_labels, args.metacaptions]):
        pipeline.extend([
            wds.map_dict(text=lambda x : filter_preprocess_txt(x, args.ds_filter, args.csv_scrambled, args.ds_cipher, args.simplecaptions, args.strict, args.shift_cipher, args.integer_labels, args.multiclass, args.metacaptions), handler=log_and_continue),
            wds.select(filter_no_caption_text),
        ])
    if args.integer_labels:
        pipeline.extend([
            wds.map_dict(image=preprocess_img, handler=log_and_continue),
        ])
    else:
        pipeline.extend([
            wds.map_dict(image=preprocess_img, text=lambda x : preprocess_txt(x, args.token_scrambled, args.token_strip), handler=log_and_continue),
        ])
    pipeline.extend([
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train),
    ])
    dataset = wds.DataPipeline(*pipeline)
    if is_train:
        if not resampled:
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)     
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True
        #collate_fn = intlabel_collate if args.integer_labels else torch.utils.data.dataloader.default_collate
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)

def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, total=None):
    collate_fn = intlabel_collate if args.integer_labels else torch.utils.data.dataloader.default_collate
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename, "No CSV filename found"
    if args.sim_clr:
        dataset = ImageAugCSVDataset(
            input_filename,
            preprocess_fn,
            img_key=args.csv_img_key,
            caption_key=args.csv_caption_key,
            csvfilter=args.ds_filter,
            csvscrambled=args.csv_scrambled,
            csvcleaned=args.csv_cleaned,
            dscipher=args.ds_cipher,
            simplecaptions=args.simplecaptions,
            strict=args.strict,
            shift=args.shift_cipher,
            integer_labels=args.integer_labels,
            metacaptions=args.metacaptions,
            sep=args.csv_separator)
    else:
        dataset = CsvDataset(
            input_filename,
            preprocess_fn,
            img_key=args.csv_img_key,
            caption_key=args.csv_caption_key,
            csvfilter=args.ds_filter,
            csvscrambled=args.csv_scrambled,
            tokenscrambled=args.token_scrambled,
            token_strip=args.token_strip,
            csvcleaned=args.csv_cleaned,
            dscipher=args.ds_cipher,
            simplecaptions=args.simplecaptions,
            strict=args.strict,
            shift=args.shift_cipher,
            integer_labels=args.integer_labels,
            multiclass=args.multiclass,
            metacaptions=args.metacaptions,
            sep=args.csv_separator)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train
        #collate_fn=collate_fn
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    return DataInfo(dataloader, sampler)

class SyntheticDataset(Dataset):

    def __init__(self, transform=None, image_size=(224, 224), caption="Dummy caption", dataset_size=100):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type in ["csv"]:
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extention {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns, epoch=0):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}
    total = TotalSize()

    if args.ds_cipher:
        args.ds_filter = get_imagenet_cipher()
    elif args.ds_filter != "":
        if args.ds_filter == "imagenet_classnames":
            args.ds_filter = get_imagenet_classnames()
        elif args.ds_filter == "imagenet_wrongorder_classnames":
            args.ds_filter = get_imagenet_wrongorder_classnames()
        elif args.ds_filter == "imagenet_our_classnames":
            args.ds_filter = get_imagenet_our_classnames()
        elif args.ds_filter == "imagenet_def_classnames":
            args.ds_filter = get_imagenet_def_classnames()
        else:
            var_names = globals()
            args.ds_filter = var_names[args.ds_filter]
            args.ds_filter = [clean_captions(a) for a in args.ds_filter]

        if args.metacaptions:
            args.metacaptions = pd.read_csv(args.metacaptions)
            args.metacaptions.fillna('', inplace=True)
        else:
            args.metacaptions = pd.DataFrame()

    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, total=total)
    elif args.schema:
        data["train"] = get_dataset_fn(args.schema, "webdataset")(
            args, preprocess_train, is_train=True, epoch=epoch, total=total) 
           
    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, total=None)
    
    if args.imagenet_train is not None:
        data["imagenet-train"] = get_imagenet(args, preprocess_fns, "train")

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    if args.imagenet_r is not None:
        data["imagenet-r"] = get_imagenet(args, preprocess_fns, "r")    
    
    if args.imagenet_s is not None:
        data["imagenet-s"] = get_imagenet(args, preprocess_fns, "s")   
    
    if args.imagenet_a is not None:
        data["imagenet-a"] = get_imagenet(args, preprocess_fns, "a")

    if args.objectnet is not None:
        data["objectnet"] = get_objectnet(args, preprocess_fns)   

    if args.inat2021 is not None:
        data["inat2021"] = get_torchvision(args, preprocess_fns, "inat2021")

    if args.stanfordcars is not None:
        data["stanfordcars"] = get_torchvision(args, preprocess_fns, "stanfordcars")
    
    if args.flowers is not None:
        data["flowers"] = get_torchvision(args, preprocess_fns, "flowers")

    if args.air is not None:
        data["air"] = get_torchvision(args, preprocess_fns, "air")

    if args.food is not None:
        data["food"] = get_torchvision(args, preprocess_fns, "food")
    if total is not None:
        data["total"] = total
    return data
