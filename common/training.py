import torch
from tqdm import tqdm
from pathlib import Path


def evaluate(model, data_loader, loss_fn, device):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)

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
):
    model.train()

    total_loss = 0.0

    if epoch is not None and num_epochs is not None:
        desc = f"Epoch {epoch}/{num_epochs}"
    else:
        desc = "Training"

    progress_bar = tqdm(
        train_loader,
        desc=desc,
        disable=not verbose,
        leave=False,
    )

    for batch_x, batch_y in progress_bar:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        outputs = model(batch_x)
        loss = loss_fn(outputs, batch_y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if verbose:
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(train_loader)


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


def load_checkpoint(path, model=None, optimizer=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)

    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


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
    print_every=1,
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

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch + 1,
            num_epochs=num_epochs,
            verbose=verbose,
        )

        val_loss = evaluate(
            model=model,
            data_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
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

        should_print = (
            verbose
            and (
                (epoch + 1) % print_every == 0
                or improved
                or epochs_without_improvement >= patience
            )
        )

        if should_print:
            status = "saved" if improved else f"no improvement {epochs_without_improvement}/{patience}"

            print(
                f"Epoch {epoch + 1:03d}/{num_epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"best_val_loss={best_val_loss:.4f} | "
                f"{status}"
            )

        if epochs_without_improvement >= patience:
            checkpoint = load_checkpoint(checkpoint_path, device=device)

            checkpoint["train_losses"] = train_losses
            checkpoint["val_losses"] = val_losses
            checkpoint["stopped_epoch"] = epoch + 1
            checkpoint["patience"] = patience

            torch.save(checkpoint, checkpoint_path)

            if verbose:
                print(f"Early stopping at epoch {epoch + 1}")

            break

    return train_losses, val_losses
