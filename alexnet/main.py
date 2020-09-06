from pl_bolts.callbacks import PrintTableMetricsCallback
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_lightning import Trainer

from alexnet.lightning_alex_net import LightningAlexNet


def main():
    model = LightningAlexNet()
    trainer = Trainer(callbacks=[PrintTableMetricsCallback()])

    cifar10 = CIFAR10DataModule()
    trainer.fit(model, cifar10)


if __name__ == "__main__":
    main()
