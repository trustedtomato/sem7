import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from models import TSEncoder
from models.losses import hierarchical_contrastive_loss
from torch.utils.data import DataLoader, TensorDataset
from utils import (
    Logger,
    centerize_vary_length_series,
    split_with_nan,
    take_per_row,
    torch_pad_nan,
)


class TS2Vec:
    """The TS2Vec model"""

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        depth,
        device,
        batch_size,
        lr=0.001,
        max_train_length=None,
        temporal_unit=0,
        after_iter_callback=None,
        after_epoch_callback=None,
    ):
        """Initialize a TS2Vec model.

        Args:
            input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
            output_dims (int): The representation dimension.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (int): The gpu used for training and inference.
            lr (int): The learning rate.
            batch_size (int): The batch size.
            max_train_length (Union[int, NoneType]): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
            temporal_unit (int): The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory.
            after_iter_callback (Union[Callable, NoneType]): A callback function that would be called after each iteration.
            after_epoch_callback (Union[Callable, NoneType]): A callback function that would be called after each epoch.
        """

        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.depth = depth

        self._net = TSEncoder(
            input_dims=self.input_dim,
            output_dims=self.output_dim,
            hidden_dims=self.hidden_dim,
            depth=self.depth,
        ).to(self.device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)

        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback

        self.n_epochs = 0

    def _train_or_val(self, data_loader, optimizer, mode):
        if mode == "train":
            self._net.train()
            self.net.train()
            torch.set_grad_enabled(True)
        else:
            self._net.eval()
            self.net.eval()
            torch.set_grad_enabled(False)

        cum_loss = 0
        n_epoch_iters = 0

        for batch in data_loader:
            x = batch[0]
            if self.max_train_length is not None and x.size(1) > self.max_train_length:
                window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                x = x[:, window_offset : window_offset + self.max_train_length]
            x = x.to(self.device)

            ts_l = x.size(1)
            crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l + 1)
            crop_left = np.random.randint(ts_l - crop_l + 1)
            crop_right = crop_left + crop_l
            crop_eleft = np.random.randint(crop_left + 1)
            crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
            crop_offset = np.random.randint(
                low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0)
            )

            if mode == "train":
                optimizer.zero_grad()

            out1 = self._net(
                take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft)
            )
            out1 = out1[:, -crop_l:]

            out2 = self._net(
                take_per_row(x, crop_offset + crop_left, crop_eright - crop_left)
            )
            out2 = out2[:, :crop_l]

            loss = hierarchical_contrastive_loss(
                out1, out2, temporal_unit=self.temporal_unit
            )
            if mode == "train":
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)

            cum_loss += loss.item()
            n_epoch_iters += 1

            if self.after_iter_callback is not None:
                self.after_iter_callback(self, loss.item())

        return cum_loss / n_epoch_iters

    def fit(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray,
        settled: int,
        folder_name: str,
        model_name: str | None = None,
        n_epochs: int | None = None,
        verbose: bool = False,
    ):
        """Training the TS2Vec model.

        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.

        Returns:
            loss_log: a list containing the training losses on each epoch.
        """
        assert train_data.ndim == 3

        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(
                    split_with_nan(train_data, sections, axis=1), axis=0
                )

        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)

        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]
        print("Creating training dataset")
        train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, len(train_dataset)),
            shuffle=True,
            drop_last=True,
        )
        print("Creating validation dataset")
        val_dataset = TensorDataset(torch.from_numpy(val_data).to(torch.float))
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(self.batch_size, len(val_dataset)),
            shuffle=True,
            drop_last=True,
        )
        # Load snapshot if it exists
        snapshot_path = f"data/{folder_name}/{model_name}_snapshot.pt"
        if os.path.exists(snapshot_path):
            snapshot = torch.load(snapshot_path, weights_only=True)
            self.net.load_state_dict(snapshot["best_averaged_state"])
            self._net.load_state_dict(snapshot["best_encoder_state"])
            self.n_epochs = snapshot["current_epoch"] + 1
            train_losses = snapshot["train_losses"]
            val_losses = snapshot["val_losses"]
            best_averaged_state = snapshot["best_averaged_state"]
            best_encoder_state = snapshot["best_encoder_state"]
        else:
            best_averaged_state = self.net.state_dict()
            best_encoder_state = self._net.state_dict()
            self.n_epochs = 0
            train_losses: list[float] = []
            val_losses: list[float] = []

        def save_snapshot(
            epoch: int, train_losses: list[float], val_losses: list[float]
        ):
            current_train_loss = train_losses[-1]
            current_val_loss = val_losses[-1]
            current_averaged_state = self.net.state_dict()
            current_encoder_state = self._net.state_dict()
            if current_train_loss <= min(train_losses) and current_val_loss <= min(
                val_losses
            ):
                nonlocal best_averaged_state
                nonlocal best_encoder_state
                best_averaged_state = current_averaged_state
                best_encoder_state = current_encoder_state

            snapshot = {
                "best_averaged_state": best_averaged_state,
                "averaged_state": current_averaged_state,
                "best_encoder_state": best_encoder_state,
                "encoder_state": current_encoder_state,
                "current_epoch": epoch,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "tsencoder_hidden_dim": self.hidden_dim,
                "tsencoder_depth": self.depth,
                "ts_embedding_dim": self.output_dim,
                "epochs": n_epochs,
            }
            torch.save(snapshot, snapshot_path)

        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        print("Starting training")
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break

            train_average_loss = self._train_or_val(
                train_loader, optimizer, mode="train"
            )
            val_average_loss = self._train_or_val(val_loader, optimizer, mode="val")
            train_losses.append(train_average_loss)
            val_losses.append(val_average_loss)

            if len(val_losses) > settled:
                global_minimum = min(val_losses)
                local_minimum = min(val_losses[-settled:])
                if local_minimum > global_minimum:
                    print(f"No improvement in {settled} epochs. Stopping training")
                    break

            save_snapshot(self.n_epochs, train_losses, val_losses)

            if verbose:
                print(
                    f"Epoch #{self.n_epochs}: train_loss={train_average_loss}, val_loss={val_average_loss}"
                )
            self.n_epochs += 1

            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, train_average_loss)

    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        out = self.net(x.to(self.device, non_blocking=True), mask)
        if encoding_window == "full_series":
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=out.size(1),
            ).transpose(1, 2)

        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=encoding_window,
                stride=1,
                padding=encoding_window // 2,
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]

        elif encoding_window == "multiscale":
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size=(1 << (p + 1)) + 1,
                    stride=1,
                    padding=1 << p,
                ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)

        else:
            if slicing is not None:
                out = out[:, slicing]

        return out.cpu()

    def encode(
        self,
        data,
        mask=None,
        encoding_window=None,
        causal=False,
        sliding_length=None,
        sliding_padding=0,
        batch_size=None,
    ):
        """Compute representations using the model.

        Args:
            data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            causal (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_length (Union[int, NoneType]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.

        Returns:
            repr: The representations for data.
        """
        assert self.net is not None, "please train or load a net first"
        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()

        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)

        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                if sliding_length is not None:
                    reprs = []
                    if n_samples < batch_size:
                        calc_buffer = []
                        calc_buffer_l = 0
                    for i in range(0, ts_l, sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not causal else 0)
                        x_sliding = torch_pad_nan(
                            x[:, max(l, 0) : min(r, ts_l)],
                            left=-l if l < 0 else 0,
                            right=r - ts_l if r > ts_l else 0,
                            dim=1,
                        )
                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out = self._eval_with_pooling(
                                    torch.cat(calc_buffer, dim=0),
                                    mask,
                                    slicing=slice(
                                        sliding_padding,
                                        sliding_padding + sliding_length,
                                    ),
                                    encoding_window=encoding_window,
                                )
                                reprs += torch.split(out, n_samples)
                                calc_buffer = []
                                calc_buffer_l = 0
                            calc_buffer.append(x_sliding)
                            calc_buffer_l += n_samples
                        else:
                            out = self._eval_with_pooling(
                                x_sliding,
                                mask,
                                slicing=slice(
                                    sliding_padding, sliding_padding + sliding_length
                                ),
                                encoding_window=encoding_window,
                            )
                            reprs.append(out)

                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out = self._eval_with_pooling(
                                torch.cat(calc_buffer, dim=0),
                                mask,
                                slicing=slice(
                                    sliding_padding, sliding_padding + sliding_length
                                ),
                                encoding_window=encoding_window,
                            )
                            reprs += torch.split(out, n_samples)
                            calc_buffer = []
                            calc_buffer_l = 0

                    out = torch.cat(reprs, dim=1)
                    if encoding_window == "full_series":
                        out = F.max_pool1d(
                            out.transpose(1, 2).contiguous(),
                            kernel_size=out.size(1),
                        ).squeeze(1)
                else:
                    out = self._eval_with_pooling(
                        x, mask, encoding_window=encoding_window
                    )
                    if encoding_window == "full_series":
                        out = out.squeeze(1)

                output.append(out)

            output = torch.cat(output, dim=0)

        self.net.train(org_training)
        return output.numpy()

    def save(self, fn):
        """Save the model to a file.

        Args:
            fn (str): filename.
        """
        torch.save(self.net.state_dict(), fn)

    def load(self, fn):
        """Load the model from a file.

        Args:
            fn (str): filename.
        """
        state_dict = torch.load(fn, map_location=self.device, weights_only=True)
        self.net.load_state_dict(state_dict)
