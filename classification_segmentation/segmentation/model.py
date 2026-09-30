"""
segmentation/model.py — TEMPLATE
=================================
Implement your semantic segmentation model here.

DO NOT rename this file or the class.
"""




# Class index mapping (pixel values in segmentation masks)
# 0 = background
# 1 = aeroplane, 2 = bicycle, 3 = bird, 4 = boat, 5 = bottle
# 6 = bus, 7 = car, 8 = cat, 9 = chair, 10 = cow
# 11 = diningtable, 12 = dog, 13 = horse, 14 = motorbike, 15 = person
# 16 = pottedplant, 17 = sheep, 18 = sofa, 19 = train, 20 = tvmonitor
# 255 = ignore / void


import os
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset
from torchvision import transforms


preprocess_seg = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def compute_iou_batch(preds, masks, num_classes=21):
    B = preds.shape[0]
    ious = []

    for b in range(B):
        iou_per_image = []
        for cls in range(num_classes):
            pred_inds = (preds[b] == cls)
            target_inds = (masks[b] == cls)

            intersection = (pred_inds & target_inds).sum().item()
            union = (pred_inds | target_inds).sum().item()

            if union == 0:
                continue
            iou_per_image.append(intersection / union)

        if len(iou_per_image) > 0:
            ious.append(sum(iou_per_image) / len(iou_per_image))

    return sum(ious) / len(ious)

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "segmentation_masks")
        self.transform = transform

        self.image_ids = sorted(os.listdir(self.image_dir))

        self.mask_resize = transforms.Resize((224, 224),interpolation=transforms.InterpolationMode.NEAREST)


    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", ".png"))

        image = np.array(Image.open(image_path).convert("RGB"))
        if self.transform:
            image = self.transform(image)

        mask = Image.open(mask_path)
        mask = self.mask_resize(mask)
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask
    



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3, padding=1,stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, kernel_size=3, stride=1,padding=1):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.block = nn.Sequential(
            ConvBlock(in_channels + skip_channels, out_channels,kernel_size=kernel_size, padding=padding, stride=stride),
            ConvBlock(out_channels, out_channels,kernel_size=kernel_size, padding=padding, stride=stride)
        )

    def forward(self, x,skip=None):
        x = self.upsample(x)

        if skip is not None:
          if x.size() != skip.size():
              x = F.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=True)
          x = torch.cat([x, skip], dim=1)

        x = self.block(x)
        return x


class MobileNetV2_Skip(nn.Module):

    def __init__(self, num_classes=21,use_skipconnections=True):
        super().__init__()

        mobilenet = models.mobilenet_v2(pretrained=True)
        self.skip = use_skipconnections

        self.encoder = mobilenet.features
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.skip_indices = [2, 4, 7, 14]  ## Indices of layers to use for skip connections
        self.bottleneck_channels = 1280

        if self.skip:
          self.skip_channels = [0,24,32,64,160]
        else:
          self.skip_channels = [0,0,0,0,0]

        self.bottleneck = ConvBlock(self.bottleneck_channels, 256)


        self.decoder4 = DecoderBlock(256, self.skip_channels[4], 192)
        self.decoder3 = DecoderBlock(192, self.skip_channels[3], 128)
        self.decoder2 = DecoderBlock(128, self.skip_channels[2], 64)
        self.decoder1 = DecoderBlock(64, self.skip_channels[1], 32)
        self.decoder0 = DecoderBlock(32, self.skip_channels[0], 32)

        # self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1,stride=1,padding=0)

        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )


        self.bottleneck.apply(self.init_weights)
        self.decoder4.apply(self.init_weights)
        self.decoder3.apply(self.init_weights)
        self.decoder2.apply(self.init_weights)
        self.decoder1.apply(self.init_weights)
        self.decoder0.apply(self.init_weights)
        self.final_conv.apply(self.init_weights)

    def init_weights(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):

        input_size = x.shape[2:]

        skip_connections = []
        for idx, layer in enumerate(self.encoder):
            x = layer(x)
            if idx in self.skip_indices:
                skip_connections.append(x)

        if self.skip:
          skip1, skip2, skip3, skip4 = skip_connections
        else:
          skip1, skip2, skip3, skip4 = None,None,None,None

        x = self.bottleneck(x)

        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        x = self.decoder0(x,None)

        x = self.final_conv(x)

        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)

        return x
    


class SegmentationModel:
    def __init__(self, weights_dir: str,model_weights: str='best_model.pth',use_skipconnections=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MobileNetV2_Skip(num_classes=21,use_skipconnections=use_skipconnections).to(self.device)

        self.weights_dir = weights_dir
        weights_path = os.path.join(weights_dir, model_weights)
     
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            print("Loaded trained segmentation weights")
        else:
            print("Using ImageNet pretrained encoder + random decoder")

        self.model.eval()
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

        self.train_epoch_loss = []
        self.val_epoch_loss = []
        self.val_epoch_iou = []


    def train(self, train_loader, val_loader, num_epochs=10, lr=1e-4):
        print("Starting segmentation training...")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0

            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)

                loss = self.criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                train_loss += loss.detach().cpu().item()

            avg_train_loss = train_loss / len(train_loader)
            print(f"[TRAIN Epoch {epoch+1}] Loss: {avg_train_loss:.4f}")

            val_loss,val_iou = self.validate(val_loader)

            self.train_epoch_loss.append(avg_train_loss)
            self.val_epoch_loss.append(val_loss)
            self.val_epoch_iou.append(val_iou)

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        val_iou = 0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validating"):
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                iou = compute_iou_batch(preds, masks)
                val_iou += iou

        avg_loss = val_loss / len(val_loader)
        avg_iou = val_iou / len(val_loader)

        print(f"[VALIDATION] Loss: {avg_loss:.4f}, mIoU: {avg_iou:.4f}")

        return avg_loss,avg_iou


    def predict(self, image: np.ndarray) -> np.ndarray:
        self.model.eval()

        H, W, _ = image.shape

        x = preprocess_seg(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(x)

        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        pred = np.array(
            Image.fromarray(pred.astype(np.uint8)).resize((W, H), Image.NEAREST),
            dtype=np.uint8
        )

        return pred

    def save_model(self, save_path: str):
        torch.save(self.model.state_dict(), os.path.join(self.weights_dir, save_path))

