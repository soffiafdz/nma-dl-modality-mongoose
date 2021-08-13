import os
import os.path as op
import torch

from collections import namedtuple
from tqdm.auto import trange

from .models import weights_init_normal


TrainResults = namedtuple(
    "TrainResults", ["model", "train_loss_history", "val_loss_history"]
)


def train_generator_only(
    model,
    optimizer,
    loss_fn,
    train_loader,
    device,
    n_epochs,
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

    device : ["cpu", "cuda"]
        The device on which to train the model

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
    model = model.to(device)
    if init_checkpoint is not None:
        checkpoint = torch.load(init_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        starting_epoch = checkpoint["epoch"] + 1
        loss = checkpoint["loss"]
        train_loss_history = checkpoint["train_loss_history"]
        val_loss_history = checkpoint["val_loss_history"]
    else:
        model.apply(weights_init_normal)
        starting_epoch = 0
        train_loss_history = []
        val_loss_history = []

    if starting_epoch > n_epochs:
        raise ValueError(
            "The checkpointed epoch number exceeds the number of requested epochs. "
            f"n_epochs = {n_epochs} but checkpointed epochs = {starting_epoch}."
        )

    epoch_range = trange(starting_epoch, n_epochs) if progress_bar else range(n_epochs)
    if checkpoint_interval is not None:
        os.makedirs(op.dirname(checkpoint_file_pattern), exist_ok=True)

    for epoch in epoch_range:
        model.train()
        for images, targets in train_loader:
            # Zero out the gradients
            optimizer.zero_grad()

            images = images.to(device)
            targets = targets.to(device)
            output = model(images)

            # Compute the loss
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()

        train_loss_history.append(loss.item())

        if val_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device)
                    targets = targets.to(device)
                    output = model(images)
                    val_loss += loss_fn(output, targets, reduction="sum").item()

            val_loss /= len(val_loader.dataset)
            val_loss_history.append(val_loss)

        if verbose_interval is not None and epoch % verbose_interval == 0:
            msg = f"Epoch {epoch:03d}: train_loss = {loss.item()}"
            if val_loader is not None:
                msg += f" val_loss = {val_loss}"
            print(msg)

        if checkpoint_interval is not None and epoch % checkpoint_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                    "train_loss_history": train_loss_history,
                    "val_loss_history": val_loss_history,
                },
                checkpoint_file_pattern % (epoch,),
            )

    return TrainResults(model, train_loss_history, val_loss_history)
