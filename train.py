#!/bin/python3

"""Train the generator."""
import os
import torch
import argparse
import numpy as np
import os.path as op

from torch.nn import ConstantPad3d
from torch.utils.data import DataLoader
from collections import namedtuple
from tqdm.auto import trange

from .models import weights_init_normal
from .models import GeneratorUNet
from .dice_loss import diceloss
from .dataset import HCPStructuralDataset

TrainResults = namedtuple(
    "TrainResults", ["model", "train_loss_history", "val_loss_history"]
)

CGANTrainResults = namedtuple(
    "CGANTrainResults",
    [
        "model",
        "total_train_loss_history",
        "g_train_loss_history",
        "d_train_loss_history",
        "val_loss_history",
    ],
)


def add_channel(img):
    """TO BE FILLED..."""
    return img.unsqueeze(0)


def pad_to_multiple_of_16(img):
    """Add padding for the image to be a multiple of 16."""
    return ConstantPad3d(padding=(0, 0, 4, 3, 0, 0), value=0)(img)


def min_max_scale(img):
    """TO BE FILLED..."""
    return 2 * (img - img.min()) / (img.max() - img.min()) - 1


def preproc(img):
    """TO BE FILLED..."""
    return add_channel(pad_to_multiple_of_16(min_max_scale(img)))


def train_generator_only(
    model,
    optimizer,
    loss_fn,
    train_loader,
    n_epochs,
    device="auto",
    val_loader=None,
    verbose_interval=None,
    progress_bar=True,
    checkpoint_interval=None,
    checkpoint_file_pattern="generator_%d.pt",
    init_checkpoint=None,
):
    """Train a generator model only.

    Parameters
    ----------
    model : torch.nn.Module subclass
        The model to train

    optimizer : torch.optim optimizer class
        The optimizer to use for training

    loss_fn : torch.nn.Module subclass
        The loss function to use for training

    train_loader : torch.utils.data.DataLoader subclass
        The dataset loader for the training data

    device : ["cpu", "cuda", "auto"]
        The device on which to train the model.
        If auto (default), use GPU (cuda) if available or fallback to cpu.

    n_epochs : int
        The number of epochs to use for training

    val_loader : torch.utils.data.DataLoader subclass, optional
        An optional dataset loader for the validation dataset. If not
        provided, validation loss will not be computed.

    verbose_interval : int, optional
        Every `verbose_interval` epochs, the train loop will print the
        training loss.

    progress_bar : bool, default=True,
        If True, show progress bar for epochs.

    checkpoint_interval : int, optional
        Every `checkpoint_interval` epochs, the train loop
        will checkpoint the model, saving it to the path in
        `checkpoint_file_pattern`.

    checkpoint_file_pattern : str, optional
        The file pattern to use to save each model checkpoint. This
        string must contain a %d token to allow the inclusion of the
        epoch number in the filename. Default = "generator_%d.pt".

    init_checkpoint : str, optional
        The path to use for the intialization checkpoint. If None,
        this function will train from scratch.
    """
    valid_devices = ["auto", "cpu", "cuda"]
    if device not in valid_devices:
        raise ValueError(
            f"device must be one of {valid_devices}. Got {device} instead."
        )
    elif device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    loss_fn = loss_fn.to(device)

    if init_checkpoint is not None:
        checkpoint = torch.load(init_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        starting_epoch = checkpoint["epoch"] + 1
        train_loss_history = checkpoint["train_loss_history"]
        val_loss_history = checkpoint["val_loss_history"]
    else:
        model.apply(weights_init_normal)
        starting_epoch = 0
        train_loss_history = []
        val_loss_history = []

    if starting_epoch > n_epochs:
        raise ValueError(
            "Checkpointed epoch number exceeds the number of requested epochs."
            f"n_epochs = {n_epochs} "
            f"but checkpointed epochs = {starting_epoch}."
        )

    epoch_range = trange(starting_epoch, n_epochs) if progress_bar else range(n_epochs)
    if checkpoint_interval is not None:
        os.makedirs(op.dirname(checkpoint_file_pattern), exist_ok=True)

    for epoch in epoch_range:
        model.train()
        train_loss = []
        for images, targets in train_loader:
            # Zero out the gradients
            optimizer.zero_grad()

            images = images.to(device)
            targets = targets.to(device)
            output = model(images)

            # Compute the loss
            loss = loss_fn(output, targets)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        train_loss = np.mean(train_loss)
        train_loss_history.append(train_loss)

        if val_loader is not None:
            model.eval()
            val_loss = []
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device)
                    targets = targets.to(device)
                    output = model(images)
                    val_loss.append(loss_fn(output, targets).item())

            val_loss = np.mean(val_loss)
            val_loss_history.append(val_loss)

        if verbose_interval is not None and epoch % verbose_interval == 0:
            msg = f"Epoch {epoch:03d}: train_loss = {train_loss}"
            if val_loader is not None:
                msg += f" val_loss = {val_loss}"
            print(msg)

        if checkpoint_interval is not None and epoch % checkpoint_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss_history": train_loss_history,
                    "val_loss_history": val_loss_history,
                },
                checkpoint_file_pattern % (epoch,),
            )

    return TrainResults(model, train_loss_history, val_loss_history)


def train_cgan(
    g_model,
    d_model,
    g_optimizer,
    d_optimizer,
    train_loader,
    n_epochs,
    d_update_threshold=0.8,
    input_image_shape=(128, 160, 128),
    lambda_voxel=100.0,
    g_loss_fn=None,
    d_loss_fn=None,
    device="auto",
    val_loader=None,
    verbose_interval=None,
    batch_verbose=False,
    progress_bar=True,
    checkpoint_interval=None,
    checkpoint_file_pattern="generator_%d.pt",
    init_checkpoint=None,
):
    """Train a generator model only.

    Parameters
    ----------
    g_model : torch.nn.Module subclass
        The generator model to train

    d_model : torch.nn.Module subclass
        The discriminator model to train

    optimizer : torch.optim optimizer class
        The optimizer to use for training

    g_loss_fn : torch.nn.Module subclass
        The loss function to use for the generator

    d_loss_fn : torch.nn.Module subclass
        The loss function to use for the discriminator

    train_loader : torch.utils.data.DataLoader subclass
        The dataset loader for the training data

    device : ["cpu", "cuda", "auto"]
        The device on which to train the model.
        If auto (default), use GPU (cuda) if available or fallback to cpu.

    n_epochs : int
        The number of epochs to use for training

    d_update_threshold : float, default=0.8
        Accuracy threshold for discriminator updating. If the discriminator
        accuracy (averaged over the patches in the "latent space") is below this
        amount, we will update the discriminator in the current batch.

    input_image_size : tuple
        Size of the input images

    lambda_voxel : float, default=100.0
        Scalar weight applied to the generator loss

    val_loader : torch.utils.data.DataLoader subclass, optional
        An optional dataset loader for the validation dataset. If not
        provided, validation loss will not be computed.

    verbose_interval : int, optional
        Every `verbose_interval` epochs, the train loop will print the
        training loss.

    batch_verbose : bool, default=False
        If True, print loss information for every batch.

    progress_bar : bool, default=True,
        If True, show progress bar for epochs.

    checkpoint_interval : int, optional
        Every `checkpoint_interval` epochs, the train loop
        will checkpoint the model, saving it to the path in
        `checkpoint_file_pattern`.

    checkpoint_file_pattern : str, optional
        The file pattern to use to save each model checkpoint. This
        string must contain a %d token to allow the inclusion of the
        epoch number in the filename. Default = "generator_%d.pt".

    init_checkpoint : str, optional
        The path to use for the intialization checkpoint. If None,
        this function will train from scratch.
    """
    valid_devices = ["auto", "cpu", "cuda"]
    if device not in valid_devices:
        raise ValueError(
            f"device must be one of {valid_devices}. Got {device} instead."
        )
    elif device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    d_loss_fn = d_loss_fn if d_loss_fn is not None else torch.nn.MSELoss()
    g_loss_fn = g_loss_fn if g_loss_fn is not None else diceloss()

    g_model = g_model.to(device)
    d_model = d_model.to(device)

    g_loss_fn = g_loss_fn.to(device)
    d_loss_fn = d_loss_fn.to(device)

    if init_checkpoint is not None:
        checkpoint = torch.load(init_checkpoint)
        g_model.load_state_dict(checkpoint["g_model_state_dict"])
        d_model.load_state_dict(checkpoint["d_model_state_dict"])
        g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
        d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
        starting_epoch = checkpoint["epoch"] + 1
        total_train_loss_history = checkpoint["total_train_loss_history"]
        g_train_loss_history = checkpoint["g_train_loss_history"]
        d_train_loss_history = checkpoint["d_train_loss_history"]
        val_loss_history = checkpoint["val_loss_history"]
    else:
        g_model.apply(weights_init_normal)
        d_model.apply(weights_init_normal)
        starting_epoch = 0
        total_train_loss_history = []
        g_train_loss_history = []
        d_train_loss_history = []
        val_loss_history = []

    if starting_epoch > n_epochs:
        raise ValueError(
            "Checkpointed epoch number exceeds the number of requested epochs."
            f"n_epochs = {n_epochs} "
            f"but checkpointed epochs = {starting_epoch}."
        )

    epoch_range = trange(starting_epoch, n_epochs) if progress_bar else range(n_epochs)
    if checkpoint_interval is not None:
        os.makedirs(op.dirname(checkpoint_file_pattern), exist_ok=True)

    patch_size = (1,) + tuple([dim // 2 ** 4 for dim in input_image_shape])

    for epoch in epoch_range:
        g_model.train()
        d_model.train()
        total_train_loss = []
        g_train_loss = []
        d_train_loss = []
        for batch_idx, images, targets in enumerate(train_loader):
            real_A = images.to(device)
            real_B = targets.to(device)
            valid = torch.ones(real_A.size(0), *patch_size, requires_grad=False).to(
                device
            )
            fake = torch.zeros(real_A.size(0), *patch_size, requires_grad=False).to(
                device
            )

            # ---------------------
            #  Train Discriminator, only update every disc_update batches
            # ---------------------
            # Real loss
            fake_B = g_model(real_A)
            pred_real = d_model(real_B, real_A)
            # print(pred_real)
            # print(valid.shape)
            loss_real = d_loss_fn(pred_real, valid)

            # Fake loss
            pred_fake = d_model(fake_B.detach(), real_A)
            loss_fake = d_loss_fn(pred_fake, fake)
            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            d_real_acu = torch.ge(pred_real.squeeze(), 0.5).float()
            d_fake_acu = torch.le(pred_fake.squeeze(), 0.5).float()
            d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu), 0))

            if d_total_acu <= d_update_threshold:
                d_optimizer.zero_grad()
                loss_D.backward()
                d_optimizer.step()
                # discriminator_update = 'True'

            # ------------------
            #  Train Generators
            # ------------------
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()

            # GAN loss
            fake_B = g_model(real_A)
            pred_fake = d_model(fake_B, real_A)
            loss_GAN = d_loss_fn(pred_fake, valid)
            # Voxel-wise loss
            loss_voxel = g_loss_fn(fake_B, real_B)

            # Total loss
            loss_G = loss_GAN + lambda_voxel * loss_voxel
            loss_G.backward()

            g_optimizer.step()

            total_train_loss.append(loss_G.item())
            d_train_loss.append(loss_GAN.item())
            g_train_loss.append(loss_voxel.item())

            if batch_verbose:
                msg = f"Epoch {epoch:03d}, batch {batch_idx}: total_loss = {loss_G.item()} "
                msg += f"G loss = {loss_voxel.item()}, D loss = {loss_GAN.item()}"
                print(msg)

        total_train_loss = np.mean(total_train_loss)
        d_train_loss = np.mean(d_train_loss)
        g_train_loss = np.mean(g_train_loss)

        total_train_loss_history.append(total_train_loss)
        g_train_loss_history.append(g_train_loss)
        d_train_loss_history.append(d_train_loss)

        if val_loader is not None:
            g_model.eval()
            val_loss = []
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device)
                    targets = targets.to(device)
                    output = g_model(images)
                    val_loss.append(g_loss_fn(output, targets).item())

            val_loss = np.mean(val_loss)
            val_loss_history.append(val_loss)

        if verbose_interval is not None and epoch % verbose_interval == 0:
            msg = f"Epoch {epoch:03d}: train_loss = {total_train_loss}"
            # if val_loader is not None:
            #     msg += f" val_loss = {val_loss}"
            print(msg)

        if checkpoint_interval is not None and epoch % checkpoint_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "g_model_state_dict": g_model.state_dict(),
                    "d_model_state_dict": d_model.state_dict(),
                    "g_optimizer_state_dict": g_optimizer.state_dict(),
                    "d_optimizer_state_dict": d_optimizer.state_dict(),
                    "total_train_loss_history": total_train_loss_history,
                    "g_train_loss_history": g_train_loss_history,
                    "d_train_loss_history": d_train_loss_history,
                    "val_loss_history": val_loss_history,
                },
                checkpoint_file_pattern % (epoch,),
            )

    return CGANTrainResults(
        model,
        total_train_loss_history,
        g_train_loss_history,
        d_train_loss_history,
        val_loss_history,
    )


if __name__ == "__main__":

    def parse_options():
        """Argument parser."""
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Train model",
        )

        parser.add_argument("data_dir", help="Study data directory")
        parser.add_argument(
            "-d",
            "--device",
            choices=["auto", "cuda", "cpu"],
            default="auto",
            help="Device on which to train the model",
        )
        parser.add_argument(
            "-w",
            "--workers",
            type=int,
            default=0,
            help="Number of workers for the DataLoader",
        )
        parser.add_argument(
            "-e",
            "--epochs",
            type=int,
            default=10,
            help="Number of epochs to use for training",
        )
        parser.add_argument("-b", "--batch", type=int, default=4, help="Batch size")
        parser.add_argument(
            "-c",
            "--checkpoint",
            type=int,
            default=1,
            help="Number of epochs after which the model will" " be saved",
        )
        parser.add_argument(
            "-i", "--init_path", help="Path to use for initialization checkpoint"
        )
        options = parser.parse_args()
        return options

    opts = parse_options()

    model = GeneratorUNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e4)
    loss = diceloss()
    train_dataloader = DataLoader(
        HCPStructuralDataset(split="train", study_dir=opts.data_dir, transform=preproc),
        batch_size=opts.batch,
        shuffle=True,
        num_workers=opts.workers,
    )
    val_dataloader = DataLoader(
        HCPStructuralDataset(
            split="validate", study_dir=opts.data_dir, transform=preproc
        ),
        batch_size=opts.batch,
        shuffle=True,
        num_workers=opts.workers,
    )
    # test_dataloader = DataLoader(
    #     HCPStructuralDataset(
    #        split="test", study_dir=opts.data_dir, transform=preproc
    #     ),
    #     batch_size=opts.batch, shuffle=True, num_workers=opts.workers
    # )

    train_generator_only(
        model=model,
        optimizer=optimizer,
        loss_fn=loss,
        train_loader=train_dataloader,
        device=opts.device,
        n_epochs=opts.epochs,
        val_loader=val_dataloader,
        verbose_interval=1,
        checkpoint_file_pattern=op.join(
            opts.data_dir, "weights", "t1_to_t2", "generator_%d.pt"
        ),
        checkpoint_interval=opts.checkpoint,
        init_checkpoint=opts.init_path,
    )
