import lightning as pl
import torch
from torch import nn
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall, MulticlassPrecision, MulticlassF1Score, MulticlassConfusionMatrix

###############################
#    Multilayer Perceptron    #
###############################

IMAGE_SIZE = 28


class CNN(pl.LightningModule):
    """
    Convolutional Neural Network (CNN) to solve MNIST with PyTorch Lightning.
    """

    def __init__(
            self,
            in_channels=28,
            out_channels=10,
            metric=[MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix],
            lr_rate=0.001,
            seed=None,
    ):
        # Set seed for reproducibility iniciialization
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        super().__init__()
        self.metric = metric
        self.lr_rate = lr_rate

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=32, kernel_size=(5, 5), padding="same"
        )
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(5, 5), padding="same"
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.l1 = nn.Linear(7 * 7 * 64, 2048)
        self.l2 = nn.Linear(2048, out_channels)

        self.loss_fn = nn.CrossEntropyLoss()
        self.training_step_outputs = []
        self.training_step_real = []

        self.validation_step_outputs = []
        self.validation_step_real = []

        self.test_step_outputs = []
        self.test_step_real = []
        self.metric=[]
        if type(metric) is list:
            for m in metric:
                self.metric.append(m(num_classes=10))
        else:
            self.metric = metric(num_classes=10)

    def forward(self, x):
        """ """
        input_layer = x.view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
        conv1 = self.relu(self.conv1(input_layer))
        pool1 = self.pool1(conv1)
        conv2 = self.relu(self.conv2(pool1))
        pool2 = self.pool2(conv2)
        pool2_flat = pool2.reshape(-1, 7 * 7 * 64)

        dense = self.relu(self.l1(pool2_flat))
        logits = self.l2(dense)

        return logits

    def configure_optimizers(self):
        """ """
        return torch.optim.Adam(self.parameters(), lr=self.lr_rate)

    def log_metrics(self, phase, y_pred, y, print_cm = True):
        if type(self.metric) is list:
            for m in self.metric:
                if (isinstance(m, MulticlassConfusionMatrix)):
                    if print_cm:
                        print(phase+"/CM\n", m(y_pred, y))
                    else:
                        pass
                else:
                    self.log(phase+"/"+m.__class__.__name__.replace("Multiclass", ""), m(y_pred, y))
        else:
            self.log(phase+"/"+self.metric.__class__.__name__.replace("Multiclass", ""), self.metric(y_pred, y))

    def training_step(self, batch, batch_id):
        """ """
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        out = torch.argmax(logits, dim=1)
        self.training_step_outputs.append(out)
        self.training_step_real.append(y)
        
        self.log("Train/Loss", loss, prog_bar=True)
        self.log_metrics("Train", out, y, print_cm=False)
        
        return loss

    def on_train_epoch_end(self):
        out = torch.cat(self.training_step_outputs)
        y = torch.cat(self.training_step_real)
        self.log_metrics("TrainEpoch", out, y, print_cm=True)

        self.training_step_outputs.clear()  # free memory
        self.training_step_real.clear()

    def validation_step(self, batch, batch_idx):
        """ """
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        out = torch.argmax(logits, dim=1)
        self.validation_step_outputs.append(out)
        self.validation_step_real.append(y)
        self.log("Validation/Loss", loss, prog_bar=True)
        self.log_metrics("Validation", out, y, print_cm=False)
        return loss
    
    def on_validation_epoch_end(self):
        out = torch.cat(self.validation_step_outputs)
        y = torch.cat(self.validation_step_real)
        self.log_metrics("ValidationEpoch", out, y, print_cm=True)

        self.validation_step_outputs.clear()  # free memory
        self.validation_step_real.clear()

    def test_step(self, batch, batch_idx):
        """ """
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        out = torch.argmax(logits, dim=1)
        self.test_step_outputs.append(out)
        self.test_step_real.append(y)
        self.log("Test/Loss", loss, prog_bar=True)
        self.log_metrics("Test", out, y, print_cm=False)
        return loss

    def on_test_epoch_end(self):
        out = torch.cat(self.test_step_outputs)
        y = torch.cat(self.test_step_real)
        self.log_metrics("TestEpoch", out, y, print_cm=True)

        self.test_step_outputs.clear()  # free memory
        self.test_step_real.clear()
