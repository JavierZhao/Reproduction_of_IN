# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import awkward as ak
import click
import h5py
import numpy as np
import uproot
import yaml

project_dir = Path(__file__).resolve().parents[2]
np.random.seed(42)


def to_np_array(ak_array, maxN=100, pad=0, dtype=float):
    """convert awkward array to regular numpy array"""
    return ak.fill_none(ak.pad_none(ak_array, maxN, clip=True, axis=-1), pad).to_numpy().astype(dtype)


@click.command()
@click.argument(
    "definitions",
    type=click.Path(exists=True),
    default=f"{project_dir}/src/data/definitions.yml",
)
@click.option("--train", is_flag=True, show_default=True, default=False)
@click.option("--test", is_flag=True, show_default=True, default=False)
@click.option("--outdir", show_default=True, default=f"{project_dir}/data/processed/")
@click.option("--max-entries", show_default=True, default=None, type=int)
@click.option("--keep-frac", show_default=True, default=1, type=float)
@click.option("--batch-size", show_default=True, default=None, type=int)
def main(definitions, train, test, outdir, max_entries, keep_frac, batch_size):  # noqa: C901
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    with open(definitions) as yaml_file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        # read in parameters from yaml file
        defn = yaml.load(yaml_file, Loader=yaml.FullLoader)

    spectators = defn["spectators"]
    labels = defn["labels"]
    n_feature_sets = defn["n_feature_sets"]
    if not batch_size:
        batch_size = defn["batch_size"]
    if train:
        dataset = "train"
    elif test:
        dataset = "test"
    else:
        logger.info("You need to specify if they are training/testing dataset by setting --train or --test")
    files = defn[f"{dataset}_files"]

    # start reading files and creating h5 files
    counter = -1
    total_entries = 0
    done = False
    for input_file in files:
        in_file = uproot.open(input_file)
        tree = in_file[defn["tree_name"]]
        nentries = tree.num_entries
        logger.info(f"opening {input_file} with {nentries} events")
        for k in range(0, nentries, batch_size):
            # create new h5 file every batch_size events
            counter += 1
            if os.path.isfile(f"{outdir}/{dataset}/newdata_{counter}.h5"):
                # skip if file already exists
                logger.info(f"{outdir}/{dataset}/newdata_{counter}.h5 exists... skipping")
                continue
            arrays = tree.arrays(spectators, library="np", entry_start=k, entry_stop=k + batch_size)
            mask = (np.random.rand(*arrays["fj_pt"].shape) < keep_frac)
            spec_array = np.expand_dims(np.stack([arrays[spec][mask] for spec in spectators], axis=1), axis=1)
            real_batch_size = spec_array.shape[0]  # real batch size might be smaller than batch_size
            total_entries += real_batch_size  # keeping count of how many entries were processed

            feature_arrays = {}  # initialize feature array dictionary (feature_name: feature array)
            for j in range(n_feature_sets):
                # initialize the feature array for the current feature
                feature_arrays[f"features_{j}"] = np.zeros(
                    (real_batch_size, defn[f"nobj_{j}"], len(defn[f"features_{j}"])),
                    dtype=float,
                )
                arrays = tree.arrays(
                    defn[f"features_{j}"],
                    entry_start=k,
                    entry_stop=k + batch_size,
                    library="ak",
                )  # feature array for the current feature
                for i, feature in enumerate(defn[f"features_{j}"]):
                    feat = to_np_array(arrays[feature][mask], maxN=defn[f"nobj_{j}"])  # turn feature into np array
                    feature_arrays[f"features_{j}"][:, :, i] = feat  # add to feature array in the dictionary
                # For PyTorch channels-first style networks
                feature_arrays[f"features_{j}"] = np.ascontiguousarray(np.swapaxes(feature_arrays[f"features_{j}"], 1, 2))

            # finished loading feature_arrays from root files within the current batch
            arrays = tree.arrays(labels, library="np", entry_start=k, entry_stop=k + batch_size)
            # target_array is the array of labels
            target_array = np.zeros((real_batch_size, 2), dtype=float)
            target_array[:, 0] = arrays["sample_isQCD"][mask] * arrays["fj_isQCD"][mask]
            target_array[:, 1] = arrays["fj_isH"][mask]

            # save the feature_arrays, target_array, and spec_array to h5 file
            os.makedirs(f"{outdir}/{dataset}", exist_ok=True)
            with h5py.File(f"{outdir}/{dataset}/newdata_{counter}.h5", "w") as h5:
                logger.info(f"creating {h5.filename} h5 file with {real_batch_size} events")
                feature_data = h5.create_group(f"{dataset}ing_subgroup")
                target_data = h5.create_group("target_subgroup")
                spec_data = h5.create_group("spectator_subgroup")
                for j in range(n_feature_sets):
                    feature_data.create_dataset(
                        f"{dataset}ing_{j}",
                        data=feature_arrays[f"features_{j}"].astype("float32"),
                    )
                    np.save(
                        f"{outdir}/{dataset}/{dataset}_{counter}_features_{j}.npy",
                        feature_arrays[f"features_{j}"].astype("float32"),
                    )  # save the features
                target_data.create_dataset("target", data=target_array.astype("float32"))
                np.save(
                    f"{outdir}/{dataset}/{dataset}_{counter}_truth.npy",
                    target_array.astype("float32"),
                )  # saving the labels
                spec_data.create_dataset("spectators", data=spec_array.astype("float32"))
                np.save(
                    f"{outdir}/{dataset}/{dataset}_{counter}_spectators.npy",
                    spec_array.astype("float32"),
                )  # saving the spectators
                h5.close()  # close the h5 file
            # check if we have reached the max number of entries
            if max_entries and total_entries >= max_entries:
                done = True
                break
        if done:
            break


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"  # format for logging
    logging.basicConfig(level=logging.INFO, format=log_fmt)  # set up logging
    main()
