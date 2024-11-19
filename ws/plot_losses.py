import os

import matplotlib.pyplot as plt


def plot_losses(
    losses_dict: list[dict],
    title: str,
    x_name: str = r"Epochs",
    y_name: str = r"Loss",
    show: bool = True,
    save_path: str = "",
    save_name: str = "losses.pdf",
):
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    # If error with type1ec.sty run apt install cm-super
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"

    figure, ax = plt.subplots(figsize=(10, 5))
    for loss_dict in losses_dict:
        train_loss = loss_dict["train_loss"]
        val_loss = loss_dict["val_loss"]
        train_label = loss_dict["train_label"]
        val_label = loss_dict["val_label"]
        train_style = loss_dict.get("train_style", "-")
        val_style = loss_dict.get("val_style", "--")
        ax.plot(
            range(1, len(train_loss) + 1), train_loss, train_style, label=train_label
        )
        ax.plot(range(1, len(val_loss) + 1), val_loss, val_style, label=val_label)
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(x_name, fontsize=14)
    ax.set_ylabel(y_name, fontsize=14)
    ax.legend(fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True)

    if save_path:
        figure.savefig(
            os.path.join(save_path, save_name),
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.05,
        )
    if show:
        plt.show()


if __name__ == "__main__":
    # Example usage
    training_losses_1 = [1.0, 0.9, 0.8, 0.7, 0.6]
    validation_losses_1 = [0.9, 0.8, 0.7, 0.6, 0.5]
    training_losses_2 = [0.8, 0.7, 0.6, 0.5, 0.4]
    validation_losses_2 = [0.7, 0.6, 0.5, 0.4, 0.3]
    losses_dict = [
        {
            "train_loss": training_losses_1,
            "train_style": "co",
            "train_label": r"Training loss 1",
            "val_loss": validation_losses_1,
            "val_style": "r+",
            "val_label": r"Validation loss 1",
        },
        {
            "train_loss": training_losses_2,
            "val_loss": validation_losses_2,
            "train_label": r"Training loss 2",
            "train_style": "bo--",
            "val_style": "g*-",
            "val_label": r"Validation loss 2",
        },
    ]

    plot_losses(
        losses_dict,
        "Training and Validation Losses",
        show=False,
        save_path="figures/",
        save_name="losses.pdf",
    )
