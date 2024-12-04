import datetime
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from ts2vec import TS2Vec


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024**2


def plot_losses(
    loss_dict: dict,
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
    train_loss = loss_dict["train_loss"]
    val_loss = loss_dict["val_loss"]
    train_label = loss_dict["train_label"]
    val_label = loss_dict["val_label"]
    train_style = loss_dict.get("train_style", "-")
    val_style = loss_dict.get("val_style", "-")
    ax.plot(range(1, len(train_loss) + 1), train_loss, train_style, label=train_label)
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
    # training_losses_1 = [1.0, 0.9, 0.8, 0.7, 0.6]
    # validation_losses_1 = [0.9, 0.8, 0.7, 0.6, 0.5]
    # training_losses_2 = [0.8, 0.7, 0.6, 0.5, 0.4]
    # validation_losses_2 = [0.7, 0.6, 0.5, 0.4, 0.3]
    # losses_dict = [
    #     {
    #         "train_loss": training_losses_1,
    #         "train_style": "co",
    #         "train_label": r"Training loss 1",
    #         "val_loss": validation_losses_1,
    #         "val_style": "r+",
    #         "val_label": r"Validation loss 1",
    #     },
    #     {
    #         "train_loss": training_losses_2,
    #         "val_loss": validation_losses_2,
    #         "train_label": r"Training loss 2",
    #         "train_style": "bo--",
    #         "val_style": "g*-",
    #         "val_label": r"Validation loss 2",
    #     },
    # ]
    # plot_losses(
    #     losses_dict,
    #     "Training and Validation Losses",
    #     show=False,
    #     save_path="figures/",
    #     save_name="losses.pdf",
    # )

    # Load all the losses from the experiments
    models_folder = "data/ts2vec"
    # Iterate over all models in folder
    loss_dicts = []
    performance_dicts = []
    for i, model_snapshot_name in enumerate(os.listdir(models_folder)):
        model_snapshot = torch.load(
            os.path.join(models_folder, model_snapshot_name),
            weights_only=True,
            map_location=torch.device("cpu"),
        )
        train_loss = model_snapshot["train_losses"]
        val_loss = model_snapshot["val_losses"]
        tsencoder_depth = model_snapshot["tsencoder_depth"]
        tsencoder_hidden_dim = model_snapshot["tsencoder_hidden_dim"]
        ts_embedding_dim = model_snapshot["ts_embedding_dim"]
        training_time_seconds = model_snapshot["training_time"]
        training_time = str(datetime.timedelta(seconds=int(training_time_seconds)))
        model_name = model_snapshot_name.replace("_snapshot.pt", "")
        model = TS2Vec(
            12,
            device="cpu",
            hidden_dim=tsencoder_hidden_dim,
            depth=tsencoder_depth,
            output_dim=ts_embedding_dim,
            batch_size=1,
        )
        model.net.load_state_dict(model_snapshot["best_averaged_state"])
        model._net.load_state_dict(model_snapshot["best_encoder_state"])
        model_size = get_model_size(model.net) + get_model_size(model._net)
        loss_dict = {
            "model_name": model_name,
            "train_loss": train_loss,
            "train_label": f"Training loss {model_name}",
            "train_style": "-",
            "val_loss": val_loss,
            "val_label": f"Validation loss {model_name}",
            "val_style": "-",
        }
        parameters = {
            "Depth": tsencoder_depth,
            "Hidden dim": tsencoder_hidden_dim,
            "Embedding dim": ts_embedding_dim,
            "Train time": training_time,
            "Min val loss": "{:.3f}".format(min(val_loss)),
            "Min train loss": "{:.3f}".format(min(train_loss)),
            "Model size": "{:.3f} MB".format(model_size),
        }
        loss_dicts.append(loss_dict)
        performance_dicts.append(parameters)
    for loss_dict in loss_dicts:
        plot_losses(
            loss_dict,
            f"Train and validation loss for model {loss_dict["model_name"]}",
            show=False,
            save_path="figures/",
            save_name=f"losses_{loss_dict["model_name"]}.pdf",
        )
        print("Done plotting")

    performance_dataframe = pd.DataFrame(performance_dicts)
    print(performance_dataframe.sort_values("Min val loss").to_latex())
