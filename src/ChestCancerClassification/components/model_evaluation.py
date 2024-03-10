import cv2
import numpy as np
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
import torch
import mlflow.pytorch
from urllib.parse import urlparse
from pathlib import Path
from ChestCancerClassification.config.configuration import EvaluationConfig
from ChestCancerClassification.utils.utils import save_json


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _compute_img_mean_std(self, image_paths):
        """
        computing the mean and std of three channel on the whole dataset,
        first we should normalize the image from 0-255 to 0-1
        """
        img_h, img_w = self.config.params_image_size, self.config.params_image_size
        imgs = []
        means, stdevs = [], []
        for i in range(len(image_paths)):
            img = cv2.imread(image_paths[i])
            img = cv2.resize(img, (img_h, img_w))
            imgs.append(img)
        imgs = np.stack(imgs, axis=3)
        print(imgs.shape)
        imgs = imgs.astype(np.float32) / 255.0
        for i in range(3):
            pixels = imgs[:, :, i, :].ravel()  # resize to one row
            means.append(np.mean(pixels))
            stdevs.append(np.std(pixels))
        means.reverse()  # BGR --> RGB
        stdevs.reverse()
        print("normMean = {}".format(means))
        print("normStd = {}".format(stdevs))
        return means, stdevs

    def _test_dataloader(self):
        norm_mean, norm_std = self._compute_img_mean_std(self.config.all_image_path)
        test_transforms = transforms.Compose(
            [
                transforms.Resize(
                    (self.config.params_image_size, self.config.params_image_size)
                ),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ]
        )
        test_dataset = datasets.ImageFolder(
            self.config.testing_data, transform=test_transforms
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.config.params_batch_size, shuffle=False
        )
        return test_dataloader

    def load_model(self, path: Path):
        model = models.vgg16(weights=self.config.params_weights)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(num_ftrs, self.config.params_classes)
        self.model = model.to(self.config.params_device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.params_learning_rate
        )
        self.criterion = torch.nn.CrossEntropyLoss().to(self.config.params_device)
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.criterion = checkpoint["loss_fn"]

    def _evalModel(self):
        self.model.eval()
        test_loss = AverageMeter()
        test_acc = AverageMeter()
        with torch.no_grad():
            for i, data in enumerate(self.test_dataloader):
                images, labels = data
                N = images.size(0)
                images = Variable(images).to(self.config.params_device)
                labels = Variable(labels).to(self.config.params_device)
                outputs = self.model(images)
                prediction = outputs.max(1, keepdim=True)[1]
                test_acc.update(
                    prediction.eq(labels.view_as(prediction)).sum().item() / N
                )
                test_loss.update(self.criterion(outputs, labels).item())
        return {
            "model_name": self.model.__class__.__name__,
            "model_loss": test_loss.avg,
            "model_acc": test_acc.avg,
        }

    def evaluation(self):
        self.load_model(self.config.path_of_model)
        self.test_dataloader = self._test_dataloader()
        self.score = self._evalModel()
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score["model_loss"], "accuracy": self.score["model_acc"]}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score["model_loss"], "accuracy": self.score["model_acc"]}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.pytorch.log_model(
                    self.model, "model", registered_model_name="VGG16Model"
                )
            else:
                mlflow.pytorch.log_model(self.model, "model")
