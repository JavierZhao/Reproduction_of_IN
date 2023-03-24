from __future__ import print_function

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# if torch.cuda.is_available():
#     import setGPU  # noqa: F401

import tqdm
import yaml

from src.data.h5data import H5Data
from src.models.gnn import GraphNet

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
project_dir = "/home/ziz078/teams/group-2/Reproduction_of_IN"

train_path = f"{project_dir}/data/processed/train/"
definitions = f"{project_dir}/src/data/definitions.yml"
with open(definitions) as yaml_file:
    defn = yaml.load(yaml_file, Loader=yaml.FullLoader)

N = defn["nobj_2"]  # number of charged particles
N_sv = defn["nobj_3"]  # number of SVs
n_targets = len(defn["reduced_labels"])  # number of classes
params = defn["features_2"]
params_sv = defn["features_3"]

model_dict = {}

device = "cuda"

files = glob.glob(os.path.join(train_path, "newdata_*.h5"))
# take first 10% of files for validation
# n_val should be 5 for full dataset
n_val = max(1, int(0.1 * len(files)))
files_val = files[:n_val]
files_train = files[n_val:]

outdir = f"{project_dir}/models/"  # output directory
vv_branch = False  # Consider vertex-vertex interaction in model
drop_rate = 0  # Signal Drop rate
load_def = False  # Load weights from default model if enabled
random_split = False  # randomly split train test data if enabled

label = "small_IN"
batch_size = 512
n_epochs = 100

model_loc = f"{outdir}/trained_models/"
model_perf_loc = f"{outdir}/model_performances"
model_dict_loc = f"{outdir}/model_dicts"
os.system(f"mkdir -p {model_loc} {model_perf_loc} {model_dict_loc}")

# Get the training and validation data
data_train = H5Data(
    batch_size=batch_size,
    cache=None,
    preloading=0,
    features_name="training_subgroup",
    labels_name="target_subgroup",
    spectators_name="spectator_subgroup",
)
data_train.set_file_names(files_train)
data_val = H5Data(
    batch_size=batch_size,
    cache=None,
    preloading=0,
    features_name="training_subgroup",
    labels_name="target_subgroup",
    spectators_name="spectator_subgroup",
)
data_val.set_file_names(files_val)

n_val = data_val.count_data()
n_train = data_train.count_data()
print(f"val data: {n_val}")
print(f"train data: {n_train}")

model = GraphNet(
        n_constituents=N,
        n_targets=n_targets,
        params=len(params),
        hidden=60,
        n_vertices=N_sv,
        params_v=len(params_sv),
        vv_branch=int(vv_branch),
        De=20,
        Do=24,
    )

optimizer = optim.Adam(model.parameters(), lr=0.0001)

loss = nn.CrossEntropyLoss(reduction="mean")

l_val_best = 99999

from sklearn.metrics import accuracy_score

softmax = torch.nn.Softmax(dim=1)
import time
loss_train_all = []
loss_val_all = []

for m in range(n_epochs):
    print(f"Epoch {m}\n")
    lst = []
    loss_val = []
    loss_train = []
    correct = []
    tic = time.perf_counter()

    # train process
    iterator = data_train.generate_data()  # needed for batch gradient descent
    total_ = int(n_train / batch_size)

    pbar = tqdm.tqdm(iterator, total=total_)  # displays progresss bar
    for element in pbar:
        (sub_X, sub_Y, _) = element
        training = sub_X[2]  # particle features
        training_sv = sub_X[3]  # secondary vertex features
        target = sub_Y[0]  # labels
        
        # convert to pytorch tensors
        trainingv = torch.tensor(training, dtype=torch.float, device=device)
        trainingv_sv = torch.tensor(training_sv, dtype=torch.float, device=device)
        targetv = (torch.from_numpy(np.argmax(target, axis=1)).long()).to(device)
        optimizer.zero_grad()
        model.train()
        
        out = model(trainingv, trainingv_sv)
        batch_loss = loss(out, targetv)
        batch_loss.backward()
        optimizer.step()
        batch_loss = batch_loss.detach().cpu().item()
        loss_train.append(batch_loss)
        pbar.set_description(f"Training loss: {batch_loss:.4f}")
    toc = time.perf_counter()
    print(f"Training done in {toc - tic:0.4f} seconds")
    tic = time.perf_counter()
    
    # validate process
    iterator = data_val.generate_data()
    total_ = int(n_val / batch_size)
    pbar = tqdm.tqdm(iterator, total=total_)
    for element in pbar:
        (sub_X, sub_Y, _) = element
        training = sub_X[2]
        training_sv = sub_X[3]
        target = sub_Y[0]

        trainingv = torch.tensor(training, dtype=torch.float, device=device)
        trainingv_sv = torch.tensor(training_sv, dtype=torch.float, device=device)
        targetv = (torch.from_numpy(np.argmax(target, axis=1)).long()).to(device)
        model.eval()
        out = model(trainingv, trainingv_sv)
        
        lst.append(softmax(out).cpu().data.numpy())
        l_val = loss(out, targetv).cpu().item()
        loss_val.append(l_val)
        correct.append(target)
        pbar.set_description(f"Validation loss: {l_val:.4f}")
    toc = time.perf_counter()
    print(f"Evaluation done in {toc - tic:0.4f} seconds")
    l_val = np.mean(np.array(loss_val))

    predicted = np.concatenate(lst)
    print(f"\nValidation Loss: {l_val}")

    l_training = np.mean(np.array(loss_train))
    print(f"Training Loss: {l_training}")
    val_targetv = np.concatenate(correct)

    loss_train_all.append(l_training)
    loss_val_all.append(l_val)
    if l_val < l_val_best:
        print("new best model")
        l_val_best = l_val
        torch.save(model.state_dict(), f"{model_loc}/{label}_best.pth")  # save the model's state dictionary
        np.save(
            f"{model_perf_loc}/{label}_validation_target_vals.npy",
            val_targetv,
        )
        np.save(
            f"{model_perf_loc}/{label}_validation_predicted_vals.npy",
            predicted,
        )
        np.save(
            f"{model_perf_loc}/{label}_loss_train.npy",
            np.array(loss_train),
        )
        np.save(
            f"{model_perf_loc}/{label}_loss_val.npy",
            np.array(loss_val),
        )

    acc_val = accuracy_score(val_targetv[:, 0], predicted[:, 0] > 0.5)
    print(f"Validation Accuracy: {acc_val}")
    np.save(
            f"{model_perf_loc}/{label}_loss_train_all.npy",
            np.array(loss_train_all),
        )
    np.save(
        f"{model_perf_loc}/{label}_loss_val_all.npy",
        np.array(loss_val_all),
    )
    