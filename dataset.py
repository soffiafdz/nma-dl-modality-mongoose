import os.path as op
import re
import torch

from glob import glob
from torch.nn import ConstantPad3d
from torch.utils.data import Dataset


def pad_to_multiple_of_16(img):
    # required_padding = [16 - dim % 16 for dim in img.shape]
    return ConstantPad3d(padding=(0, 0, 4, 3, 0, 0), value=0)(img)


def min_max_scale(img):
    return (
        2
        * (img - img.amin(dim=(1, 2, 3)).reshape(-1, 1, 1, 1))
        / (
            img.amax(dim=(1, 2, 3)).reshape(-1, 1, 1, 1)
            - img.amin(dim=(1, 2, 3)).reshape(-1, 1, 1, 1)
        )
        - 1
    )


def default_preprocessing(img):
    return pad_to_multiple_of_16(min_max_scale(img))


class HCPDataset(Dataset):
    def __init__(
        self,
        study_dir,
        feature_transform=None,
        target_transform=None,
        input_modalities=None,
        output_modality=None,
        dwi_input_slice_idx=None,
        dwi_output_slice_idx=None,
        split="train",
        include_bval_bvec=False,
    ):
        self.include_bval_bvec = include_bval_bvec
        if self.include_bval_bvec:
            raise NotImplementedError(
                "Inclusion of the bval and bvec tensors is not currently supported."
            )

        valid_modalities = ["t1w", "t2w", "resting_lr", "resting_rl", "dwi"]

        if output_modality not in valid_modalities:
            raise ValueError(f"Output modality must be in {valid_modalities}")

        self.output_modality = (
            "dwi_predict" if output_modality == "dwi" else output_modality
        )

        self.input_modalities = (
            ["t1w"] if input_modalities is None else list(input_modalities)
        )
        if not all(modality in valid_modalities for modality in self.input_modalities):
            raise ValueError(f"All input modalities must be in {valid_modalities}")

        self.input_modalities = [
            modality.replace("dwi", "dwi_known") for modality in self.input_modalities
        ]

        valid_splits = ["train", "validate", "test"]
        if split not in valid_splits:
            raise ValueError(
                f"split must be on of {valid_splits}. Got {split} instead."
            )

        sub_regex = re.compile("sub-[0-9]*")
        self.study_dir = study_dir
        subjects = sorted(
            [
                sub_regex.search(path).group()
                for path in glob(op.join(study_dir, "sub-*"))
            ]
        )

        self.subjects = [
            subject
            for subject in subjects
            if (
                op.exists(op.join(study_dir, subject, "rfMRI_1RL.pt"))
                and op.exists(op.join(study_dir, subject, "rfMRI_1LR.pt"))
            )
        ]

        n_subjects = len(self.subjects)
        n_train = int(n_subjects * 0.6)
        n_validate = int(n_subjects * 0.2)

        if split == "train":
            self.subjects = self.subjects[:n_train]
        elif split == "validate":
            self.subjects = self.subjects[n_train : n_train + n_validate]
        else:
            self.subjects = self.subjects[n_train + n_validate :]

        modality2filename_map = {
            "t1w": "t1w.pt",
            "t2w": "t2w.pt",
            "resting_lr": "rfMRI_1LR.pt",
            "resting_rl": "rfMRI_1RL.pt",
            "dwi_known": "dwi_known.pt",
            "dwi_predict": "dwi_predict.pt",
            "bval_known": "bval_predict.pt",
            "bval_predict": "bval_predict.pt",
            "bvec_known": "bvec_known.pt",
            "bvec_predict": "bvec_predict.pt",
        }

        self.input_paths = {}
        for modality in self.input_modalities:
            self.input_paths[modality] = [
                op.join(study_dir, sub, modality2filename_map[modality])
                for sub in self.subjects
            ]
            if modality == "dwi_known" and self.include_bval_bvec:
                for filename in ["bval_known", "bvec_known"]:
                    self.input_paths[filename] = [
                        op.join(study_dir, sub, modality2filename_map[filename])
                        for sub in self.subjects
                    ]

        self.output_paths = {
            self.output_modality: [
                op.join(study_dir, sub, modality2filename_map[self.output_modality])
                for sub in self.subjects
            ]
        }
        if self.output_modality == "dwi_predict" and self.include_bval_bvec:
            for filename in ["bval_predict", "bvec_predict"]:
                self.output_paths[filename] = [
                    op.join(study_dir, sub, modality2filename_map[filename])
                    for sub in self.subjects
                ]

        self.feature_transform = (
            default_preprocessing
            if feature_transform == "default"
            else feature_transform
        )
        self.target_transform = (
            default_preprocessing if target_transform == "default" else target_transform
        )

        self.dwi_input_slice_idx = dwi_input_slice_idx
        self.dwi_output_slice_idx = dwi_output_slice_idx

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        input_images = []
        for modality, paths in self.input_paths.items():
            img = torch.load(paths[idx])

            # Either add a channel or if DWI, move channel to first dimension
            if "dwi" in modality:
                img = img.permute(3, 0, 1, 2)
                if self.dwi_input_slice_idx is not None:
                    indices = torch.tensor(self.dwi_input_slice_idx)
                    img = torch.index_select(img, 0, indices)
            else:
                img = img.unsqueeze(0)

            if self.feature_transform:
                img = self.feature_transform(img)

            input_images.append(img)

        input_img = torch.cat(input_images).float()
        output_img = torch.load(self.output_paths[self.output_modality][idx]).float()

        # Either add a channel or if DWI, move channel to first dimension
        if "dwi" in self.output_modality:
            output_img = output_img.permute(3, 0, 1, 2)
            if self.dwi_output_slice_idx is not None:
                indices = torch.tensor(self.dwi_output_slice_idx)
                output_img = torch.index_select(output_img, 0, indices)
        else:
            output_img = output_img.unsqueeze(0)

        if self.target_transform:
            output_img = self.target_transform(output_img)

        return input_img, output_img


class HCPStructuralDataset(Dataset):
    def __init__(
        self,
        study_dir,
        transform=None,
        direction="t1w_to_t2w",
        split="train",
        slice_dir=None,
        slices=None,
    ):
        valid_directions = ["t1w_to_t2w", "t2w_to_t1w"]
        if direction not in valid_directions:
            raise ValueError(
                f"direction must be one of {valid_directions}."
                f"Got {direction} instead."
            )
        self.direction = direction

        valid_slice_dirs = ["x", "y", "z"]
        max_slices = {"x": 153, "y": 128, "z": 128}
        if slice_dir is not None:
            if slice_dir not in valid_slice_dirs:
                raise ValueError(
                    f"direction must be one of {valid_slice_dirs}."
                    f"Got {slice_dir} instead."
                )

            if not all([isinstance(s, int) for s in slices]):
                raise TypeError(
                    "slice must be of type int."
                    f"Got {[type(s) for s in slices]} instead"
                )

            # Directions [y, x, z]
            if not all([s < max_slices[slice_dir] for s in slices]):
                raise ValueError(
                    "slice is out of bounds."
                    f"The max slice in {slice_dir} is {max_slices[slice_dir]}."
                )

        self.slices = slices
        self.n_slices = len(slices) if slices is not None else None
        self.slice_dir = slice_dir

        valid_splits = ["train", "validate", "test"]
        if split not in valid_splits:
            raise ValueError(
                f"split must be on of {valid_splits}. Got {split} instead."
            )

        sub_regex = re.compile("sub-[0-9]*")
        self.study_dir = study_dir
        self.subjects = sorted(
            [
                sub_regex.search(path).group()
                for path in glob(op.join(study_dir, "sub-*"))
            ]
        )

        n_subjects = len(self.subjects)
        n_train = int(n_subjects * 0.6)
        n_validate = int(n_subjects * 0.2)

        if split == "train":
            self.subjects = self.subjects[:n_train]
        elif split == "validate":
            self.subjects = self.subjects[n_train : n_train + n_validate]
        else:
            self.subjects = self.subjects[n_train + n_validate :]

        self.t1w_paths = [op.join(study_dir, sub, "t1w.pt") for sub in self.subjects]
        self.t2w_paths = [op.join(study_dir, sub, "t2w.pt") for sub in self.subjects]
        self.transform = transform

    def __len__(self):
        if self.slices is None:
            return len(self.subjects)
        else:
            return len(self.subjects) * len(self.slices)

    def __getitem__(self, idx):
        if self.slices is None:
            t1w_img = torch.load(self.t1w_paths[idx])
            t2w_img = torch.load(self.t2w_paths[idx])
        else:
            n_slices = self.n_slices
            t1w_img = torch.load(self.t1w_paths[idx // n_slices])
            t2w_img = torch.load(self.t2w_paths[idx // n_slices])
            if self.slice_dir == "x":
                t1w_img = t1w_img[self.slices[idx % n_slices], :, :]
                t2w_img = t2w_img[self.slices[idx % n_slices], :, :]
            elif self.slice_dir == "y":
                t1w_img = t1w_img[:, self.slices[idx % n_slices], :]
                t2w_img = t2w_img[:, self.slices[idx % n_slices], :]
            else:
                t1w_img = t1w_img[:, :, self.slices[idx % n_slices]]
                t2w_img = t2w_img[:, :, self.slices[idx % n_slices]]

        if self.transform:
            t1w_img = self.transform(t1w_img)
            t2w_img = self.transform(t2w_img)

        if self.direction == "t1w_to_t2w":
            return t1w_img, t2w_img
        else:
            return t2w_img, t1w_img
