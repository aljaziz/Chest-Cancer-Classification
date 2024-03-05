from ChestCancerClassification.config.configuration import PrepareBaseModelConfig
from pathlib import Path
from torchvision import models
import torch.nn as nn
import torch


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig) -> None:
        self.config = config

    def get_base_model(self):
        self.model = models.densenet121(weights=self.config.params_weights)
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(
        model, classes, freeze_all, freeze_till, learning_rate, device
    ):
        if freeze_all:
            for param in model.parameters():
                param.requires_grad = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for i in range(freeze_till):
                for param in model.features[i].parameters():
                    model.requires_grad = False
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, classes)
        full_model = model.to(device)
        optimizer = torch.optim.Adam(full_model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss().to(device)
        print(full_model)
        return {
            "model_state_dict": full_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_fn": criterion,
        }

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=False,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate,
            device=self.config.params_device,
        )
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model):
        torch.save(model, path)
