from pathlib import Path
import json
from retry import retry
from tqdm.auto import tqdm

cn_path = Path("/scratch/bf996/vlhub/metadata/dc_classnames.json")

cupl_path = Path("/scratch/bf996/vlhub/metadata/cupl_prompts.json")

with open(cn_path, "r") as cn_file:
    cn_dict = json.load(cn_file)

with open(cupl_path, "r") as cupl_file:
    cupl_dict = json.load(cupl_file)

all_classnames = []
for k, v in cn_dict.items():
    all_classnames.extend(v)

classnames_without_imagenet = []
for k, v in cn_dict.items():
    if "imagenet" not in k:
        classnames_without_imagenet.extend(v)

all_classnames = list(set(classnames_without_imagenet))

print("Classnames as a set")
print(len(all_classnames))

import openai
import os
from dotenv import load_dotenv
load_dotenv("/scratch/bf996/notebooks/.env")
openai.api_key = os.getenv("OPENAI_API_KEY")
assert openai.api_key != None, "api key did not load"

@retry(Exception, tries=3, delay=3)
def get_gpt_resp(prompt):
    return openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt},
    ],
    temperature=0,
    ).choices[0]['message']['content']

update_dict = {
    "non-imagenet": {},
}

def get_prompts(classname):
    return [
        f"Please provide a brief description of {classname}. \n",
        f"Tell me about the appearance of {classname}. \n",
        f"Tell me some facts about this thing: {classname}. \n",
        f"Describe {classname} as you would for someone who was blind. Be brief. \n",
        f"What kind of thing is this? {classname} \n",
        f"Tell me some facts that would help me identify this: {classname}. \n",
        f"Describe {classname} to me as if I were a child. \n",
        f"You are the author of an encyclopedia. Please write a brief entry for {classname}. \n",
    ]

for classname in tqdm(all_classnames):
    if update_dict["non-imagenet"].get(classname, -1) == -1:
        update_dict["non-imagenet"][classname] = [f"A photo of the {classname}."]
        prompts = get_prompts(classname)
        for prompt in prompts:
            update_dict["non-imagenet"][classname].append(get_gpt_resp(prompt))
    cupl_dict.update(update_dict)
    with open(cupl_path, "w") as cupl_file:
        json.dump(cupl_dict, cupl_file)
    

