import open_clip

print(open_clip.__file__)

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')

torch.save(model.state_dict(),"/vast/km3888/arbor_evals/bioclip.pth")


def get_preprocess():
    with open("/home/km3888/open_clip/transform.pkl", "rb") as f:
        loaded_transforms = pickle.load(f)
    loaded_transforms = pickle.load(f)
    loaded_transform = transforms.Compose(loaded_transforms)
    return loaded_transform