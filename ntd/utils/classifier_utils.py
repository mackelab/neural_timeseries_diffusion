import hydra
import numpy as np
import torch
import wandb
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn, optim
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class BinaryTensorDataset(Dataset):
    """
    Given two data tensors, creates a binary classification dataset
    with data from the first tensor as control and data from the second tensor as target.
    """

    def __init__(self, data_tensor_control, data_tensor_target):
        super().__init__()
        self.data_tensor_control = data_tensor_control
        self.data_size_control = len(data_tensor_control)
        self.data_tensor_target = data_tensor_target
        self.data_size_target = len(data_tensor_target)

    # Control is 0.0, Target is 1.0
    def __getitem__(self, index):
        if index < self.data_size_control:
            return {
                "signal": self.data_tensor_control[index],
                "label": torch.tensor([np.float32(0.0)]),
            }
        else:
            return {
                "signal": self.data_tensor_target[index - self.data_size_control],
                "label": torch.tensor([np.float32(1.0)]),
            }

    def __len__(self):
        return self.data_size_control + self.data_size_target


class TensorLabelDataset(Dataset):
    """
    Given a data tensor and a label tensor, creates a minimal classification dataset.
    """

    def __init__(self, data_tensor, label_tensor):
        assert len(data_tensor) == len(label_tensor)
        self.data_tensor = data_tensor
        self.label_tensor = label_tensor

    def __getitem__(self, index):
        return {
            "signal": self.data_tensor[index],
            "label": self.label_tensor[index],
        }

    def __len__(self):
        return len(self.data_tensor)


class GlobalAveragePoolingClassifier(nn.Module):
    """
    Time series classifier based on the classifier from Wang et al. (2019).
    Uses global average pooling (GAP) in the last layer.
    """

    def __init__(
        self,
        in_channel=1,
        hidden_channel=20,
        kernel_size=29,
        dilation=1,
    ):
        super().__init__()
        self.device = "cpu"
        self.in_channel = in_channel
        self.hidden_channel = hidden_channel
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 == 1
        self.dilation = dilation
        self.padding = (self.dilation * (self.kernel_size - 1)) // 2

        self.conv_pool = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_channel,
                out_channels=self.hidden_channel,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                padding=self.padding,
            ),
            nn.BatchNorm1d(self.hidden_channel),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.hidden_channel,
                out_channels=self.hidden_channel,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                padding=self.padding,
            ),
            nn.BatchNorm1d(self.hidden_channel),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.hidden_channel,
                out_channels=self.hidden_channel,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                padding=self.padding,
            ),
            nn.BatchNorm1d(self.hidden_channel),
            nn.ReLU(),
        )
        self.final_conv = nn.Conv1d(
            in_channels=self.hidden_channel,
            out_channels=self.hidden_channel,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
        )
        self.lin_comb = nn.Linear(self.hidden_channel, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sig):
        temp = self.conv_pool(sig)
        last_act = self.final_conv(temp)
        downpool = torch.mean(last_act, dim=2)
        return self.sigmoid(self.lin_comb(downpool))

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device = args[0]
        return self


def train_classifier(classifier, data_loader, criterion, optimizer):
    """
    Train a classifier for one epoch.

    Args:
        classifier: Classifier to train.
        data_loader: Data loader for training.
        criterion: Loss function.
        optimizer: Optimizer.

    Returns:
        Overall loss
        Array of classification labels
        Array of classification scores
    """
    classifier.train()

    agg_loss = 0.0
    concat_ys = []
    concat_scores = []
    for batch in data_loader:
        xs = batch["signal"]
        ys = batch["label"]
        xs = xs.to(classifier.device)
        ys = ys.to(classifier.device)

        scores = classifier.forward(xs)
        loss = criterion(scores, ys)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = xs.shape[0]
        agg_loss += loss.item() * batch_size
        concat_ys.append(ys.detach().cpu())
        concat_scores.append(scores.detach().cpu())

    return agg_loss, torch.cat(concat_ys, dim=0), torch.cat(concat_scores, dim=0)


def test_classifier(classifier, data_loader, criterion):
    """
    Evaluate a classifier.

    Args:
        classifier: Classifier to evaluate.
        data_loader: Data loader for evaluation.
        criterion: Loss function.

    Returns:
        Overall loss
        Array of classification labels
        Array of classification scores
    """

    classifier.eval()

    agg_loss = 0.0
    concat_ys = []
    concat_scores = []
    for batch in data_loader:
        xs = batch["signal"]
        ys = batch["label"]
        xs = xs.to(classifier.device)
        ys = ys.to(classifier.device)

        with torch.no_grad():
            scores = classifier.forward(xs)
            loss = criterion(scores, ys)

        batch_size = xs.shape[0]
        agg_loss += loss.item() * batch_size
        concat_ys.append(ys.detach().cpu())
        concat_scores.append(scores.detach().cpu())

    return agg_loss, torch.cat(concat_ys, dim=0), torch.cat(concat_scores, dim=0)


def train_and_test_classifier(
    train_dataset,
    test_dataset,
    in_channel,
    hidden_channel,
    kernel_size,
    num_runs,
    train_batch_size,
    test_batch_size,
    learning_rate,
    weight_decay,
    epochs,
    wandb_mode,
    wandb_experiment,
    wandb_tag,
    wandb_project,
    wandb_entity,
    wandb_dir,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Train and evaluate the GAP classifier given a train and test dataset.

    Args:
        train_dataset: Training dataset.
        test_dataset: Test dataset.
        in_channel: Number of input channels of the GAP classifier.
        kernel_size: Kernel size of the GAP classifier.
        num_runs: Number of runs to train and evaluate the GAP classifier.
        train_batch_size: Batch size for training.
        test_batch_size: Batch size for testing.
        learning_rate: Learning rate for training.
        weight_decay: Weight decay for training.
        epochs: Number of epochs for training.
        wandb_mode: Mode for wandb.
        wandb_experiment: Experiment name for wandb.
        wandb_tag: Tag for wandb.
        wandb_project: Project name for wandb.
        wandb_entity: Entity name for wandb.
        wandb_dir: Directory for wandb.
        device: Device to use for training and evaluation.

    Returns:
        Trained classifier
        Dictionary of test classification labels and scores for each run
    """

    final_test_dict = {}
    models_dict = {}
    for run_id in range(num_runs):
        run = wandb.init(
            mode=wandb_mode,
            project=wandb_project,
            entity=wandb_entity,
            dir=wandb_dir,
            group=wandb_experiment,
            name=f"{wandb_tag}_classifier_run_{run_id}",
        )

        classifier = GlobalAveragePoolingClassifier(
            in_channel=in_channel,
            hidden_channel=hidden_channel,
            kernel_size=kernel_size,
        )
        classifier = classifier.to(device)

        run.summary["train_data_size"] = len(train_dataset)
        run.summary["test_data_size"] = len(test_dataset)
        train_loader = DataLoader(train_dataset, train_batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, test_batch_size, shuffle=True)

        criterion = BCELoss(reduction="mean")
        optimizer = optim.AdamW(
            classifier.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        for epoch in range(epochs):
            test_loss, test_labs, test_scores = test_classifier(
                classifier, test_loader, criterion
            )
            test_labs = test_labs.numpy()
            test_scores = test_scores.numpy()
            test_metrics = {
                "agg_test_loss": test_loss / len(test_dataset),
                "agg_test_acc": accuracy_score(np.round(test_scores), test_labs),
                "agg_test_auc": roc_auc_score(test_labs, test_scores),
            }
            wandb.log(test_metrics, commit=False)

            train_loss, train_labs, train_scores = train_classifier(
                classifier, train_loader, criterion, optimizer
            )
            train_labs = train_labs.numpy()
            train_scores = train_scores.numpy()
            train_metrics = {
                "agg_train_loss": train_loss / len(train_dataset),
                "agg_train_acc": accuracy_score(np.round(train_scores), train_labs),
                "agg_train_auc": roc_auc_score(train_labs, train_scores),
            }
            wandb.log(train_metrics)

        _, final_test_labs, final_test_scores = test_classifier(
            classifier, test_loader, criterion
        )

        final_test_dict[run_id] = {
            "labels": final_test_labs,
            "scores": final_test_scores,
        }
        models_dict[run_id] = classifier.state_dict()

        run.finish()

    return classifier, final_test_dict


if __name__ == "__main__":
    from ntd.train_diffusion_model import init_dataset

    @hydra.main(version_base=None, config_path="../../conf", config_name="config")
    def main(cfg):
        train_dataset, test_dataset = init_dataset(cfg)
        train_and_test_classifier(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            in_channel=cfg.classifier.in_channel,
            hidden_channel=cfg.classifier.hidden_channel,
            kernel_size=cfg.classifier.kernel_size,
            num_runs=cfg.classifier.num_runs,
            train_batch_size=cfg.classifier.train_batch_size,
            test_batch_size=cfg.classifier.test_batch_size,
            learning_rate=cfg.classifier.learning_rate,
            weight_decay=cfg.classifier.weight_decay,
            epochs=cfg.classifier.epochs,
            wandb_mode=cfg.base.wandb_mode,
            wandb_experiment=cfg.base.experiment,
            wandb_tag=cfg.base.tag,
            wandb_project=cfg.base.wandb_project,
            wandb_entity=cfg.base.wandb_entity,
            wandb_dir=cfg.base.home_path,
        )

    main()
