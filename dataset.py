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
