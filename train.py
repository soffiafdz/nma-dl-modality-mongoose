import os
import os.path as op
import torch

from tqdm.auto import trange


def train_generator_only(
    model,
    optimizer,
    train_loader,
    loss_fn,
    device,
    n_epochs,
    verbose_interval=None,
    progress=True,
    checkpoint_interval=None,
    checkpoint_file_pattern="generator_%d.pt",
    init_checkpoint=None,
):
    model = model.to(device)
    if init_checkpoint is not None:
        checkpoint = torch.load(init_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        starting_epoch = checkpoint["epoch"] + 1
        loss = checkpoint["loss"]
    else:
        starting_epoch = 0

    if starting_epoch > n_epochs:
        raise ValueError(
            "The checkpointed epoch number exceeds the number of requested epochs. "
            f"n_epochs = {n_epochs} but checkpointed epochs = {starting_epoch}."
        )

    epoch_range = trange(starting_epoch, n_epochs) if progress else range(n_epochs)
    os.makedirs(op.dirname(checkpoint_file_pattern), exist_ok=True)

    model.train()
    for epoch in epoch_range:
        for images, targets in train_loader:
            # Zero out the gradients
            optimizer.zero_grad()

            images = images.to(device)
            targets = targets.to(device)

            # Predict a T2w image
            output = model(images)

            # Compute the dice loss
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()

        if verbose_interval is not None and epoch % verbose_interval == 0:
            print(f"Epoch {epoch:03d}: loss = {loss.item()}")

        if checkpoint_interval is not None and epoch % checkpoint_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                },
                checkpoint_file_pattern % (epoch,),
            )

    return model
