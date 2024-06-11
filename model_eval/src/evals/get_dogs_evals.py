import timm
import sys
sys.path.append('/scratch/bf996/vlhub/src')
from pathlib import Path
import os
import subprocess

# from sklearn.metrics import confusion_matrix
import pandas as pd

from tqdm import tqdm
tqdm.pandas()

from get_in100_evals import image_size_set

metrics_path = Path("/scratch/bf996/caption-paper-ICLR/combined_robustness_metrics_in100_dogs.csv")
metrics_df = pd.read_csv(metrics_path)
models = timm.list_models()
metrics_names = metrics_df['name'].tolist()
os.chdir("/scratch/bf996/vlhub")
save_csv = "/scratch/bf996/caption-paper-ICLR/combined_robustness_metrics_in100_dogs.csv"
val_p = "/imagenet/val/"
v2_p = "/scratch/bf996/datasets"
s_p = "/imagenet-sketch"
a_p = "/imagenet-a"
r_p = "/imagenet-r"
term = "in100_dogs"
for model_name in tqdm(models):
    if model_name in metrics_names:
        continue
    image_size = str(image_size_set(model_name))
    cmd = 'python src/training/main.py --batch-size=32 --workers=8 --save-results-to-csv={} --imagenet-val {} --imagenet-v2 {} --imagenet-s {} --imagenet-a {} --imagenet-r {} --model={} --zeroshot-frequency=1 --caption-subset={} --image-size={} --linear-probe=True;'.format(save_csv, val_p, v2_p, s_p, a_p, r_p, model_name, term, image_size)
    subprocess.run(cmd.split(" "))