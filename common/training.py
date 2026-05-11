import torch
from tqdm.notebook import tqdm
from pathlib import Path

# Compute loss functions for different types of models (standard vs normalizing flow)


def compute_standard_loss(model, batch_x, batch_y, loss_fn):
    """
    For normal models where:
        outputs = model(batch_x)
        loss = loss_fn(outputs, batch_y)
    """
    outputs = model(batch_x)
    loss = loss_fn(outputs, batch_y)
    return loss


def compute_flow_loss(model, batch_x, batch_y, loss_fn):
    """
    For normalizing-flow models where the loss needs access to:
        model, batch_x, batch_y
    """
    loss = loss_fn(model, batch_x, batch_y)
    return loss


def update_training_progress(
    progress_bar,
    train_loss,
    val_loss,
    best_val_loss,
    improved,
    epochs_without_improvement,
    patience,
    show_loss_improvement=False,
):
    postfix = {
        "train_loss": f"{train_loss:.4f}",
        "val_loss": f"{val_loss:.4f}",
        "best": f"{best_val_loss:.4f}",
        "patience": f"{epochs_without_improvement}/{patience}",
    }

    if show_loss_improvement:
        postfix["status"] = "saved" if improved else "no improvement"

    progress_bar.set_postfix(postfix)
    progress_bar.update(1)


def evaluate(model, data_loader, loss_fn, device, compute_loss_fn=compute_standard_loss):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            loss = compute_loss_fn(model, batch_x, batch_y, loss_fn)

            total_loss += loss.item()

    return total_loss / len(data_loader)


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    loss_fn,
    device,
    epoch=None,
    num_epochs=None,
    verbose=True,
    compute_loss_fn=compute_standard_loss,
):
    model.train()

    total_loss = 0.0

    if epoch is not None and num_epochs is not None:
        desc = f"Epoch {epoch}/{num_epochs}"
    else:
        desc = "Training"

    batch_iterator = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{num_epochs}",
        leave=False,
        disable=not verbose,
        dynamic_ncols=True,
        mininterval=1.0,
        position=1,
    )

    for batch_x, batch_y in batch_iterator:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        loss = compute_loss_fn(model, batch_x, batch_y, loss_fn)

        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        total_loss += loss_value

        if verbose:
            batch_iterator.set_postfix(loss=f"{loss_value:.4f}")

    return total_loss / len(train_loader)


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    num_epochs,
    checkpoint_path,
    patience=10,
    skip_training=False,
    extra_checkpoint_info=None,
    verbose=True,
    compute_loss_fn=compute_standard_loss,
    show_epoch_progress=True,
    show_loss_improvement=False,
    leave_model_progress=False,
):
    if skip_training:
        if verbose:
            print("Skipping training loop as skip_training is set to True.")

        checkpoint = load_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            device=device,
        )

        return checkpoint["train_losses"], checkpoint["val_losses"]

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    stopped_epoch = None

    epoch_progress = tqdm(
        total=num_epochs,
        desc="Training model",
        disable=not verbose,
        leave=leave_model_progress,
        dynamic_ncols=True,
        mininterval=1.0,
        position=0,
    )

    try:
        for epoch in range(num_epochs):
            train_loss = train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                epoch=epoch + 1,
                num_epochs=num_epochs,
                verbose=verbose and show_epoch_progress,
                compute_loss_fn=compute_loss_fn,
            )

            val_loss = evaluate(
                model=model,
                data_loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                compute_loss_fn=compute_loss_fn,
            )

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            improved = val_loss < best_val_loss

            if improved:
                best_val_loss = val_loss
                epochs_without_improvement = 0

                save_checkpoint(
                    path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    train_losses=train_losses,
                    val_losses=val_losses,
                    best_val_loss=best_val_loss,
                    extra=extra_checkpoint_info,
                )
            else:
                epochs_without_improvement += 1

            update_training_progress(
                progress_bar=epoch_progress,
                train_loss=train_loss,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                improved=improved,
                epochs_without_improvement=epochs_without_improvement,
                patience=patience,
                show_loss_improvement=show_loss_improvement,
            )

            if epochs_without_improvement >= patience:
                stopped_epoch = epoch + 1

                checkpoint = load_checkpoint(checkpoint_path, device=device)

                checkpoint["train_losses"] = train_losses
                checkpoint["val_losses"] = val_losses
                checkpoint["stopped_epoch"] = stopped_epoch
                checkpoint["patience"] = patience

                torch.save(checkpoint, checkpoint_path)

                if verbose:
                    tqdm.write(
                        f"Early stopping at epoch {stopped_epoch}/{num_epochs} | "
                        f"best_val_loss={best_val_loss:.4f}"
                    )

                break

    finally:
        epoch_progress.close()

    if verbose:
        final_epoch = stopped_epoch if stopped_epoch is not None else num_epochs

        tqdm.write(
            f"Training completed | "
            f"epochs={final_epoch}/{num_epochs} | "
            f"best_val_loss={best_val_loss:.4f}"
        )

    return train_losses, val_losses


def load_checkpoint(path, model=None, optimizer=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)

    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    train_losses,
    val_losses,
    best_val_loss,
    extra=None,
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
    }

    if extra is not None:
        checkpoint.update(extra)

    torch.save(checkpoint, path)
