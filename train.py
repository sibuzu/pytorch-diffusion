import torch
from data import DiffSet
import pytorch_lightning as pl
from model import DiffusionModel
from torch.utils.data import DataLoader
from torchsummary import summary
import imageio
import glob
import matplotlib.pyplot as plt
import argparse
import re
import os

# Training hyperparameters
diffusion_steps = 1000
dataset_choice = "CIFAR"
batch_size = 256
log_path = "./train_logs"

def load_data(dsname):
    dataset = DiffSet(True, dsname)
    testset = DiffSet(False, dsname)
    print(f"dataset size = {dataset.dataset_len}")
    print(f"testset size = {testset.dataset_len}")
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    return dataset, testset, loader

def get_checkpoint(dsname, ver):
    print(f"{log_path}/{dsname}/version_{ver}/checkpoints/*.ckpt")
    files = sorted(glob.glob(f"{log_path}/{dsname}/version_{ver}/checkpoints/*.ckpt"))
    if len(files) > 0:
        chkpoint = files[-1]
        pattern = r"epoch=(\d+)"
        # Find the match using the pattern
        match = re.search(pattern, chkpoint)
        epoch_number = int(match.group(1)) if match else 0
        return chkpoint, epoch_number
    else:
        return "", 0
    
def load_model(chk_point, sz, depth):
    if os.path.isfile(chk_point):
        model = DiffusionModel.load_from_checkpoint(chk_point, 
            in_size = sz * sz,
            t_range = diffusion_steps, 
            img_depth = depth)
    else:
        model = DiffusionModel(sz * sz, diffusion_steps, depth)
    model = model.cuda()
    return model

def get_samples(dataset, count):
    return [dataset[i] for i in range(count)]

def cat_images(samples, nw, nh):
    stacks = [torch.cat(samples[i:i+nw], dim=2) for i in range(0, nw*nh, nw)]
    total = torch.cat(stacks, dim=1)
    total = total.moveaxis(0, -1)
    total = (total + 1) / 2
    return total

def main():
    parser = argparse.ArgumentParser(description="diffusion training")

    # Add arguments to the parser
    parser.add_argument("-d", "--dsname", type=str, default="CIFAR")
    parser.add_argument("-n", "--epoch", type=int, default=50)
    parser.add_argument("-v", "--version", type=int, default=1)

    # Parse the input arguments
    args = parser.parse_args()
    
    dataset, testset, loader = load_data(args.dsname)
    chk_point, epoch = get_checkpoint(args.dsname, args.version)
    model = load_model(chk_point, dataset.size, dataset.depth)
    summary(model, [(dataset.depth, dataset.size, dataset.size), (1,)])

    samples = get_samples(testset, 4*3)
    img = cat_images(samples, 4, 3)
    print(samples[0].shape)
    print(img.shape)
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    main()
    exit(0)

# Loading parameters
load_model = None
load_version_num = 1

# Code for optionally loading model
pass_version = None
last_checkpoint = None

if load_model:
    pass_version = load_version_num
    last_checkpoint = glob.glob(
        f"./lightning_logs/{dataset_choice}/version_{load_version_num}/checkpoints/*.ckpt"
    )[-1]

# Create datasets and data loaders
train_dataset = DiffSet(True, dataset_choice)
val_dataset = DiffSet(False, dataset_choice)

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

# Create model and trainer
if load_model:
    model = DiffusionModel.load_from_checkpoint(last_checkpoint, 
        in_size=train_dataset.size*train_dataset.size, 
        t_range=diffusion_steps, img_depth=train_dataset.depth)
else:
    model = DiffusionModel(train_dataset.size*train_dataset.size, diffusion_steps, train_dataset.depth)

# print(model)
# print()
input_shape = train_dataset[0].shape
summary(model, input_shape)
print()