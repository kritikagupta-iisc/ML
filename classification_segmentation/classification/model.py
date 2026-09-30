"""
classification/model.py — TEMPLATE
====================================
Implement your multilabel classification model here.

DO NOT rename this file or the class.
"""
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
import torchvision.models as models
from torchmetrics.classification import MulticlassF1Score
from torch.utils.data import Dataset, DataLoader

cls_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class MultiLabelDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")
        self.transform = transform

        self.labels_df = pd.read_csv(os.path.join(root_dir, "labels.csv"))
        self.image_ids = self.labels_df["image_id"].values
        self.labels = self.labels_df.iloc[:, 1:].values

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx] + ".jpg"
        img_path = os.path.join(self.image_dir, img_name)

        image = np.array(Image.open(img_path).convert("RGB"))
        label = self.labels[idx].astype(np.float32)

        if self.transform:
            image = self.transform(image)

        return image, label
    


class MobileNetV2MultiLabel(nn.Module):
    def __init__(self,num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.mobilenet_base = models.mobilenet_v2(pretrained=True)

        for param in self.mobilenet_base.features.parameters():
            param.requires_grad = False

        in_features = self.mobilenet_base.classifier[1].in_features
        dim = 512 ## can change this dimension  to 256
        self.mobilenet_base.classifier = nn.Sequential(
            nn.Linear(in_features, dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(dim, self.num_classes),
        )

    def forward(self, x):
        return self.mobilenet_base(x)
    



class ClassificationModel:
    def __init__(self, weights_dir: str,model_weights: str = "best_model.pth"):
        """
        Initialize your model and load trained weights.

        Args:
            weights_dir: absolute path to classification/weights/ folder
                         where your saved model files are stored.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MobileNetV2MultiLabel()
        self.model.to(self.device)

        self.weights_dir = weights_dir
        weights_path = os.path.join(weights_dir, model_weights)
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            print("Loaded trained weights")
        else:
            for m in self.model.mobilenet_base.classifier:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
            print("Using ImageNet pretrained backbone + random classifier")

        self.model.eval()
        self.criterion = nn.BCEWithLogitsLoss()
        self.f1_metric = MulticlassF1Score(num_classes=21,average="macro",ignore_index=255).to(self.device)

        self.class_names = [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]
        self.train_epoch_loss = []
        self.val_epoch_loss = []
        self.val_epoch_acc = []

    def train(self, train_loader, val_loader, num_epochs=10, lr=1e-4,threshold=0.5):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        print("Starting training...")

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0

            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
                images = images.to(self.device,non_blocking=True)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()

                train_loss += loss.detach().cpu().item()

            avg_train_loss = train_loss / len(train_loader)
            print(f"[TRAINING Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")

            val_loss,val_acc = self.validate(val_loader,threshold=threshold)

            self.train_epoch_loss.append(avg_train_loss)
            self.val_epoch_loss.append(val_loss)
            self.val_epoch_acc.append(val_acc)


    def validate(self, val_loader, threshold=0.5):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        self.f1_metric.reset()

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels.float())
                val_loss += loss.item()

                probs = torch.sigmoid(outputs)
                preds = (probs >= threshold).float()

                correct += (preds == labels).sum().item()
                total += labels.numel()

                self.f1_metric.update(preds, labels)

        avg_loss = val_loss / len(val_loader)
        accuracy = correct / total
        f1_score = self.f1_metric.compute()

        print(f"[VALIDATION] Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f} | F1 Score: {f1_score:.4f}")

        return avg_loss, accuracy


    def predict(self, image: np.ndarray) -> dict:
        """
        Predict which of the 20 classes are present in the image.

        Args:
            image: RGB image as numpy array, shape (H, W, 3), dtype uint8

        Returns:
            dict mapping class_name (str) -> probability (float in [0, 1])
            Must contain ALL 20 classes listed in dataset_info.json.

        Example output:
            {
                "aeroplane": 0.02, "bicycle": 0.95, "bird": 0.01,
                "boat": 0.03, "bottle": 0.10, "bus": 0.00,
                "car": 0.12, "cat": 0.01, "chair": 0.04,
                "cow": 0.00, "diningtable": 0.01, "dog": 0.02,
                "horse": 0.00, "motorbike": 0.03, "person": 0.88,
                "pottedplant": 0.01, "sheep": 0.00, "sofa": 0.02,
                "train": 0.01, "tvmonitor": 0.05
            }
        """
        self.model.eval()

        image = cls_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image)

        probs = torch.sigmoid(outputs).cpu().numpy()[0]
        return {class_name: float(prob) for class_name, prob in zip(self.class_names, probs) }

    def save_model(self, save_path: str):
        torch.save(self.model.state_dict(), os.path.join(self.weights_dir , save_path))

