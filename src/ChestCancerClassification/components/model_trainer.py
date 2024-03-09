import cv2
import numpy as np
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from tqdm import tqdm
from pathlib import Path
from ChestCancerClassification.entity.config_entity import TrainingConfig


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


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
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

    def _prepare_transforms(self):
        norm_mean, norm_std = self._compute_img_mean_std(self.config.all_image_path)
        train_transforms = transforms.Compose(
            [
                transforms.Resize(
                    (self.config.params_image_size, self.config.params_image_size)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ]
        )
        test_transforms = transforms.Compose(
            [
                transforms.Resize(
                    (self.config.params_image_size, self.config.params_image_size)
                ),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ]
        )
        return train_transforms, test_transforms

    def train_valid_dataloader(self):
        train_transforms, valid_transforms = self._prepare_transforms()
        train_dataset = datasets.ImageFolder(
            self.config.training_data, transform=train_transforms
        )
        valid_dataset = datasets.ImageFolder(
            self.config.validation_data, transform=valid_transforms
        )
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=self.config.params_batch_size, shuffle=True
        )
        self.val_dataloader = DataLoader(
            valid_dataset, batch_size=self.config.params_batch_size, shuffle=False
        )

    @staticmethod
    def save_model(path: Path, model):
        torch.save(model, path)

    def _train(self, epoch):
        total_loss_train, total_acc_train = [], []
        self.model.train()
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        curr_iter = (self.config.params_epochs - 1) * len(self.train_dataloader)
        for i, data in enumerate(self.train_dataloader):
            images, labels = data
            N = images.size(0)
            # print('image shape:',images.size(0), 'label shape',labels.size(0))
            images = Variable(images).to(self.config.params_device)
            labels = Variable(labels).to(self.config.params_device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            prediction = outputs.max(1, keepdim=True)[1]
            train_acc.update(prediction.eq(labels.view_as(prediction)).sum().item() / N)
            train_loss.update(loss.item())
            curr_iter += 1
        print("------------------------------------------------------------")
        print(
            "[epoch %d], [train loss %.5f], [train acc %.5f]"
            % (epoch, train_loss.avg, train_acc.avg)
        )
        print("------------------------------------------------------------")
        total_loss_train.append(train_loss.avg)
        total_acc_train.append(train_acc.avg)
        return train_loss.avg, train_acc.avg

    def _validate(self, epoch):
        self.model.eval()
        val_loss = AverageMeter()
        val_acc = AverageMeter()
        with torch.no_grad():
            for i, data in enumerate(self.val_dataloader):
                images, labels = data
                N = images.size(0)
                images = Variable(images).to(self.config.params_device)
                labels = Variable(labels).to(self.config.params_device)
                outputs = self.model(images)
                prediction = outputs.max(1, keepdim=True)[1]
                val_acc.update(
                    prediction.eq(labels.view_as(prediction)).sum().item() / N
                )
                val_loss.update(self.criterion(outputs, labels).item())

        print("------------------------------------------------------------")
        print(
            "[epoch %d], [val loss %.5f], [val acc %.5f]"
            % (epoch, val_loss.avg, val_acc.avg)
        )
        print("------------------------------------------------------------")
        return val_loss.avg, val_acc.avg

    def fit(self):
        total_loss_val, total_acc_val = [], []
        best_val_acc = 0
        for epoch in tqdm(range(self.config.params_epochs)):
            loss_train, acc_train = self._train(epoch)
            loss_val, acc_val = self._validate(epoch)
            total_loss_val.append(loss_val)
            total_acc_val.append(acc_val)
            if acc_val > best_val_acc:
                best_val_acc = acc_val
                print("*****************************************************")
                print(
                    "best record: [epoch %d], [val loss %.5f], [val acc %.5f]"
                    % (epoch, loss_val, acc_val)
                )
                print("*****************************************************")
        self.save_model(
            path=self.config.trained_model_path,
            model={
                "epoch": self.config.params_epochs,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss_fn": self.criterion,
            },
        )
