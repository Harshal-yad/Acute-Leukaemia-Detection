import os
import re
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.linear_model import LogisticRegression


# =====================================================================
# 1. REPRODUCIBILITY & SEED
# =====================================================================
def set_seed(seed=42):
    """Sets random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# =====================================================================
# 2. DATASET INGESTION & PARSING
# =====================================================================
def load_and_combine_datasets(idb1_dir="../ALL_IDB1/im", idb2_png_dir="../ALL_IDB2/img_png"):
    """
    Parses and unifies ALL_IDB1 and ALL_IDB2 image directories into a single DataFrame.
    Gracefully resolves relative paths whether called from root or subfolders.
    Labels: 0 = Non-Blast (Healthy), 1 = Blast (ALL)
    """
    # Check if directories exist directly or in parent dir
    if not os.path.exists(idb1_dir):
        if os.path.exists("ALL_IDB1/im"):
            idb1_dir = "ALL_IDB1/im"
            idb2_png_dir = "ALL_IDB2/img_png"

    data = []
    # Process ALL_IDB1
    if os.path.exists(idb1_dir):
        for f in os.listdir(idb1_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                match = re.search(r'_([01])\.', f)
                if match:
                    label = int(match.group(1))
                    data.append({
                        "filepath": os.path.abspath(os.path.join(idb1_dir, f)),
                        "filename": f,
                        "dataset": "ALL_IDB1",
                        "label": label,
                        "class_name": "Blast (ALL)" if label == 1 else "Non-Blast (Healthy)"
                    })

    # Process ALL_IDB2
    if os.path.exists(idb2_png_dir):
        for f in os.listdir(idb2_png_dir):
            if f.lower().endswith(('.png', '.jpg')):
                match = re.search(r'_([01])\.', f)
                if match:
                    label = int(match.group(1))
                    data.append({
                        "filepath": os.path.abspath(os.path.join(idb2_png_dir, f)),
                        "filename": f,
                        "dataset": "ALL_IDB2",
                        "label": label,
                        "class_name": "Blast (ALL)" if label == 1 else "Non-Blast (Healthy)"
                    })

    df = pd.DataFrame(data)
    print(f"[Dataset] Loaded {len(df)} images total (ALL_IDB1: {(df['dataset']=='ALL_IDB1').sum()}, ALL_IDB2: {(df['dataset']=='ALL_IDB2').sum()}).")
    return df


# =====================================================================
# 3. PHASE 1: STAIN NORMALIZATION (MACENKO & REINHARD)
# =====================================================================
class MacenkoNormalizer:
    """
    Macenko Stain Normalization calibrates Hematoxylin & Eosin / Giemsa optical density
    across different laboratories and imaging conditions using SVD decomposition.
    """
    def __init__(self, alpha=1.0, beta=0.15):
        self.alpha = alpha
        self.beta = beta
        # Reference H&E stain vectors and maximum concentrations (standard reference)
        self.HERef = np.array([
            [0.5626, 0.2159],
            [0.7201, 0.8012],
            [0.4062, 0.5581]
        ])
        self.maxCRef = np.array([1.9705, 1.0308])

    def fit_transform(self, img_np):
        """
        Takes uint8 or float RGB image (H, W, 3) and returns stain normalized RGB image (H, W, 3) in [0, 255] uint8.
        """
        if img_np.dtype != np.uint8:
            img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

        h, w, c = img_np.shape
        img = img_np.reshape((-1, 3)).astype(np.float64)

        # 1. Optical Density (OD) calculation
        OD = -np.log10((img + 1.0) / 255.0)

        # 2. Remove transparent/background pixels (OD < beta)
        ODhat = OD[~np.any(OD < self.beta, axis=1)]
        if len(ODhat) < 10:
            return img_np  # Fallback if too few cellular pixels

        # 3. SVD decomposition on OD covariance
        _, _, V = np.linalg.svd(ODhat, full_matrices=False)
        # Project on plane of the two largest singular vectors
        That = np.dot(ODhat, V[0:2, :].T)

        # 4. Find extreme angles (alpha and 100 - alpha percentiles)
        phi = np.arctan2(That[:, 1], That[:, 0])
        minPhi = np.percentile(phi, self.alpha)
        maxPhi = np.percentile(phi, 100 - self.alpha)

        vMin = np.dot(V[0:2, :].T, np.array([np.cos(minPhi), np.sin(minPhi)]))
        vMax = np.dot(V[0:2, :].T, np.array([np.cos(maxPhi), np.sin(maxPhi)]))

        # First component is Hematoxylin (darker/bluer), second is Eosin/Giemsa
        if vMin[0] > vMax[0]:
            HE = np.array([vMin, vMax]).T
        else:
            HE = np.array([vMax, vMin]).T

        # 5. Determine concentrations of individual stains
        Y = np.reshape(OD, (-1, 3)).T
        C, _, _, _ = np.linalg.lstsq(HE, Y, rcond=None)

        # Normalize stain concentrations
        maxC = np.percentile(C, 99, axis=1)
        tmp = np.divide(maxC, self.maxCRef, out=np.zeros_like(maxC), where=self.maxCRef != 0)
        C2 = np.divide(C, tmp[:, np.newaxis], out=np.zeros_like(C), where=tmp[:, np.newaxis] != 0)

        # 6. Recreate image with reference stain vectors
        reconstructed = np.multiply(255.0, 10.0 ** (-np.dot(self.HERef, C2)))
        reconstructed = np.clip(reconstructed.T, 0, 255).astype(np.uint8)
        return reconstructed.reshape((h, w, c))


class ReinhardNormalizer:
    """
    Reinhard Normalization transfers color characteristics in the Lab color space.
    Fast and robust against slide preparation variance.
    """
    def __init__(self):
        # Target reference mean and standard deviation from standard Giemsa lymphoblast
        self.target_means = np.array([148.5, 142.1, 120.3])
        self.target_stds = np.array([32.4, 18.2, 14.5])

    def fit_transform(self, img_np):
        if img_np.dtype != np.uint8:
            img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

        # Convert to float
        src = img_np.astype(np.float32)
        src_means = np.mean(src, axis=(0, 1))
        src_stds = np.std(src, axis=(0, 1)) + 1e-6

        # Scale and shift
        norm = (src - src_means) * (self.target_stds / src_stds) + self.target_means
        return np.clip(norm, 0, 255).astype(np.uint8)


# =====================================================================
# 4. PYTORCH DATASET WITH STAIN NORMALIZATION & AUGMENTATION
# =====================================================================
class StainNormalizedLeukemiaDataset(Dataset):
    """
    PyTorch Dataset with optional Macenko/Reinhard stain normalization
    and 224x224 cytological augmentations.
    """
    def __init__(self, dataframe, transform=None, normalizer=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.normalizer = normalizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")

        if self.normalizer is not None:
            img_np = np.array(image)
            try:
                norm_np = self.normalizer.fit_transform(img_np)
                image = Image.fromarray(norm_np)
            except Exception:
                pass  # Fallback to original image if SVD fails on low-variance border

        label = row["label"]
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_advanced_transforms(img_size=224):
    """
    Returns enhanced 224x224 training and evaluation transforms
    preserving fine nuclear chromatin textures.
    """
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transforms, eval_transforms


def create_advanced_dataloaders(df, batch_size=16, img_size=224, test_size=0.30, val_ratio_of_test=0.50, normalizer_type="reinhard"):
    """
    Performs stratified train/val/test split and returns PyTorch DataLoaders
    equipped with chosen Stain Normalizer.
    """
    normalizer = None
    if normalizer_type == "macenko":
        normalizer = MacenkoNormalizer()
    elif normalizer_type == "reinhard":
        normalizer = ReinhardNormalizer()

    train_df, temp_df = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=val_ratio_of_test, random_state=42, stratify=temp_df["label"]
    )

    train_tf, eval_tf = get_advanced_transforms(img_size)
    train_ds = StainNormalizedLeukemiaDataset(train_df, transform=train_tf, normalizer=normalizer)
    val_ds = StainNormalizedLeukemiaDataset(val_df, transform=eval_tf, normalizer=normalizer)
    test_ds = StainNormalizedLeukemiaDataset(test_df, transform=eval_tf, normalizer=normalizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[Advanced Splits] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)} (Normalizer: {normalizer_type})")
    return train_loader, val_loader, test_loader, test_ds


def denormalize(tensor):
    """Denormalizes ImageNet standardized tensor back to [0, 1] range for visualization."""
    inv_mean = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]
    inv_std = [1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225]
    inv_norm = transforms.Normalize(mean=inv_mean, std=inv_std)
    return inv_norm(tensor).clamp(0, 1)


# =====================================================================
# 5. PHASE 3: CBAM (CONVOLUTIONAL BLOCK ATTENTION MODULE)
# =====================================================================
class ChannelAttention(nn.Module):
    """Channel Attention component of CBAM focusing on informative feature channels."""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        reduced_planes = max(in_planes // ratio, 8)
        self.fc1 = nn.Conv2d(in_planes, reduced_planes, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention component of CBAM focusing on blast nuclei vs. RBC background."""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """Convolutional Block Attention Module combining channel and spatial attention."""
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# =====================================================================
# 6. PHASE 2: ARCFACE (ADDITIVE ANGULAR MARGIN LOSS) & SUPCON
# =====================================================================
class ArcMarginProduct(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss module.
    Projects L2-normalized embeddings onto hypersphere and enforces an angular margin 'm'.
    """
    def __init__(self, in_features, num_classes=2, s=30.0, m=0.35, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # 1. Normalize input features and weight vectors
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        if label is None:
            # During evaluation, return scaled cosine directly
            return cosine * self.s

        # 2. Compute cos(theta + m)
        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 3. Apply one-hot margin offset
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss for representation learning."""
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, dim=1)
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Compute cosine similarity logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        loss = -mean_log_prob_pos.mean()
        return loss


# =====================================================================
# 7. UNIFIED MODEL ARCHITECTURE (EFFICIENTNET_V2_S + CBAM + ARCFACE)
# =====================================================================
class LeukemiaAttentionModel(nn.Module):
    """
    Advanced cytology model with Fused-MBConv EfficientNetV2 (or ConvNeXt/ResNet50),
    CBAM multi-scale attention, 512-d L2 projection, and ArcFace angular margin classifier.
    """
    def __init__(self, backbone_name="efficientnet_v2_s", num_classes=2, embedding_dim=512,
                 arcface_m=0.35, arcface_s=30.0, use_cbam=True):
        super(LeukemiaAttentionModel, self).__init__()
        self.backbone_name = backbone_name.lower()
        self.use_cbam = use_cbam

        if self.backbone_name == "efficientnet_v2_s":
            base_model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
            self.features = base_model.features
            in_features = 1280
            self.target_conv_layer = self.features[-1]
        elif self.backbone_name == "convnext_tiny":
            base_model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
            self.features = base_model.features
            in_features = 768
            self.target_conv_layer = self.features[-1]
        elif self.backbone_name == "resnet50":
            base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.features = nn.Sequential(
                base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool,
                base_model.layer1, base_model.layer2, base_model.layer3, base_model.layer4
            )
            in_features = 2048
            self.target_conv_layer = base_model.layer4[-1]
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Attention layer
        if self.use_cbam:
            self.cbam = CBAM(in_planes=in_features)
        else:
            self.cbam = nn.Identity()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Embedding projection head
        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.PReLU()
        )

        # ArcFace margin classifier
        self.arcface = ArcMarginProduct(
            in_features=embedding_dim,
            num_classes=num_classes,
            s=arcface_s,
            m=arcface_m
        )

    def extract_features(self, x):
        """Extracts spatial conv feature maps."""
        return self.features(x)

    def get_embeddings(self, x):
        """Returns L2-normalized embeddings."""
        feat = self.extract_features(x)
        feat = self.cbam(feat)
        pooled = self.pool(feat)
        embed = self.embedding_head(pooled)
        return F.normalize(embed, p=2, dim=1)

    def forward(self, x, labels=None):
        """Forward pass through backbone, CBAM, embedding head, and ArcFace."""
        feat = self.extract_features(x)
        feat = self.cbam(feat)
        pooled = self.pool(feat)
        embed = self.embedding_head(pooled)
        logits = self.arcface(embed, labels)
        return logits, embed


# =====================================================================
# 8. STAGED FINE-TUNING TRAINING PIPELINE
# =====================================================================
def train_advanced_pipeline(model, train_loader, val_loader, device,
                            freeze_epochs=10, unfreeze_epochs=15,
                            lr_head=1e-3, lr_backbone=1e-5,
                            checkpoint_name="best_advanced_efficientnet_v2.pth"):
    """
    Two-stage training procedure:
    - Stage 1 (Epochs 1 to freeze_epochs): Backbone frozen; trains CBAM + ArcFace head at lr_head.
    - Stage 2 (Epochs freeze_epochs+1 to total): Unfreezes top conv blocks at low lr_backbone.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    total_epochs = freeze_epochs + unfreeze_epochs

    print(f"\n[Stage 1: Frozen Backbone] Training CBAM & ArcFace Head for {freeze_epochs} epochs...")
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW([
        {'params': model.cbam.parameters(), 'lr': lr_head},
        {'params': model.embedding_head.parameters(), 'lr': lr_head},
        {'params': model.arcface.parameters(), 'lr': lr_head}
    ], weight_decay=1e-4)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, total_epochs + 1):
        if epoch == freeze_epochs + 1:
            print(f"\n[Stage 2: Differential Fine-Tuning] Unfreezing top backbone blocks (lr={lr_backbone})...")
            # Unfreeze top layers of backbone
            if hasattr(model.features, '__len__'):
                # For sequential feature backbones like EfficientNet or ConvNeXt
                num_blocks = len(model.features)
                unfreeze_from = max(num_blocks - 3, 0)
                for idx in range(unfreeze_from, num_blocks):
                    for param in model.features[idx].parameters():
                        param.requires_grad = True
            else:
                for param in model.features.parameters():
                    param.requires_grad = True

            optimizer = optim.AdamW([
                {'params': filter(lambda p: p.requires_grad, model.features.parameters()), 'lr': lr_backbone},
                {'params': model.cbam.parameters(), 'lr': lr_head * 0.2},
                {'params': model.embedding_head.parameters(), 'lr': lr_head * 0.2},
                {'params': model.arcface.parameters(), 'lr': lr_head * 0.2}
            ], weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=unfreeze_epochs)

        # Training Step
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits, _ = model(inputs, targets)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(logits, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation Step
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits, _ = model(inputs, None)
                loss = criterion(logits, targets)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(logits, 1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        if epoch > freeze_epochs:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{total_epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_name)
            print(f"  [Checkpoint] Saved best model to {checkpoint_name} (Val Acc: {val_acc*100:.2f}%)")

    # Load best weights
    if os.path.exists(checkpoint_name):
        model.load_state_dict(torch.load(checkpoint_name, map_location=device))

    return model, history


# =====================================================================
# 9. PHASE 4: PROBABILISTIC & OOD CALIBRATION WITH UNCERTAINTY REJECTION
# =====================================================================
class CalibratedInference:
    """
    Fits logistic probability calibration over cosine margins and implements
    an Uncertainty Rejection Threshold (tau) routing ambiguous slides to pathologist review.
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.calibrator = LogisticRegression(C=1.0)
        self.is_fitted = False

    def fit_calibration(self, dataloader):
        """Extracts validation cosine logits and fits Platt temperature scaling."""
        self.model.eval()
        all_logits, all_targets = [], []
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                logits, _ = self.model(inputs, None)
                margin = (logits[:, 1] - logits[:, 0]).cpu().numpy()
                all_logits.extend(margin)
                all_targets.extend(targets.numpy())

        X = np.array(all_logits).reshape(-1, 1)
        y = np.array(all_targets)
        self.calibrator.fit(X, y)
        self.is_fitted = True
        print("[Calibration] Logistic probability calibration fitted successfully.")

    def predict_with_rejection(self, dataloader, tau=0.80):
        """
        Infers probabilities with an uncertainty margin threshold (tau).
        Returns categorized decisions: 1 (Blast), 0 (Healthy), or -1 (Flagged for Pathologist Review).
        """
        self.model.eval()
        results = []
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                logits, _ = self.model(inputs, None)
                margin = (logits[:, 1] - logits[:, 0]).cpu().numpy()

                if self.is_fitted:
                    probs = self.calibrator.predict_proba(margin.reshape(-1, 1))
                else:
                    probs = F.softmax(logits, dim=1).cpu().numpy()

                for prob, target in zip(probs, targets.numpy()):
                    p_healthy, p_blast = prob[0], prob[1]
                    max_conf = max(p_healthy, p_blast)

                    if max_conf >= tau:
                        decision = 1 if p_blast > p_healthy else 0
                        status = "Automated"
                    else:
                        decision = -1  # Rejected for pathologist review
                        status = "Flagged: Pathologist Review"

                    results.append({
                        "p_healthy": p_healthy,
                        "p_blast": p_blast,
                        "confidence": max_conf,
                        "decision": decision,
                        "target": target,
                        "status": status
                    })

        df_res = pd.DataFrame(results)
        coverage = (df_res["decision"] != -1).mean() * 100.0
        decided = df_res[df_res["decision"] != -1]
        acc_decided = accuracy_score(decided["target"], decided["decision"]) * 100.0 if len(decided) > 0 else 0.0

        print(f"\n{'='*20} UNCERTAINTY REJECTION ANALYSIS (tau={tau}) {'='*20}")
        print(f"Total Evaluated: {len(df_res)} slides")
        print(f"Automated Coverage: {coverage:.1f}% ({len(decided)} slides)")
        print(f"Flagged for Pathologist Review: {100.0 - coverage:.1f}% ({(df_res['decision'] == -1).sum()} slides)")
        print(f"Accuracy on Automated Cases: {acc_decided:.2f}%")

        return df_res


# =====================================================================
# 10. PHASE 5: GRAD-CAM++ FOR MULTI-INSTANCE CYTOLOGY LOCALIZATION
# =====================================================================
class GradCAMPlusPlus:
    """
    Grad-CAM++ module computing higher-order partial derivatives
    to capture multi-instance lymphoblasts and discrete nucleolar chromatin clumps.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        for param in self.model.parameters():
            param.requires_grad = True

        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, output):
        self.activations = output

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate_heatmap(self, input_tensor, target_class=None):
        self.model.eval()
        logits, _ = self.model(input_tensor, None)

        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()

        score = logits[:, target_class]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        grads = self.gradients
        acts = self.activations

        # Grad-CAM++ higher order derivatives
        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads

        sum_acts = torch.sum(acts, dim=[2, 3], keepdim=True)
        eps = 1e-7
        aij = grads_power_2 / (2 * grads_power_2 + sum_acts * grads_power_3 + eps)
        aij = torch.where(grads != 0, aij, torch.zeros_like(aij))

        weights = torch.sum(aij * torch.relu(grads), dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)

        cam = cam.squeeze().cpu().detach().numpy()
        cam_min, cam_max = np.min(cam), np.max(cam)
        cam = (cam - cam_min) / (cam_max - cam_min + eps)
        confidence = F.softmax(logits, dim=1)[0, target_class].item()

        return cam, target_class, confidence

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def overlay_gradcam(original_img_np, heatmap, alpha=0.45):
    """Overlays heatmap on original RGB image."""
    cmap = plt.get_cmap('jet')
    heatmap_colored = cmap(heatmap)[:, :, :3]
    overlay = (1.0 - alpha) * original_img_np + alpha * heatmap_colored
    return np.clip(overlay, 0, 1), heatmap_colored


def visualize_gradcampp_gallery(model, target_layer, dataset, device, num_samples=4):
    """Visualizes Grad-CAM++ attributions comparing True Positives and True Negatives."""
    cam_engine = GradCAMPlusPlus(model, target_layer)
    indices = []

    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if label.item() == 1 and len([i for i in indices if dataset[i][1].item() == 1]) < num_samples // 2:
            indices.append(idx)
        elif label.item() == 0 and len([i for i in indices if dataset[i][1].item() == 0]) < num_samples // 2:
            indices.append(idx)
        if len(indices) == num_samples:
            break

    fig, axes = plt.subplots(num_samples, 3, figsize=(11, 3.2 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, sample_idx in enumerate(indices):
        img_tensor, true_label = dataset[sample_idx]
        input_batch = img_tensor.unsqueeze(0).to(device)
        heatmap, pred_class, confidence = cam_engine.generate_heatmap(input_batch)

        orig_np = denormalize(img_tensor).permute(1, 2, 0).numpy()
        overlay, _ = overlay_gradcam(orig_np, heatmap)

        true_txt = "Blast (ALL)" if true_label.item() == 1 else "Healthy (Non-Blast)"
        pred_txt = "Blast (ALL)" if pred_class == 1 else "Healthy (Non-Blast)"
        status_col = "green" if pred_class == true_label.item() else "red"

        axes[row, 0].imshow(orig_np)
        axes[row, 0].set_title(f"Stain Calibrated | True: {true_txt}", fontsize=9)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(heatmap, cmap="jet")
        axes[row, 1].set_title(f"Grad-CAM++ | Pred: {pred_txt} ({confidence*100:.1f}%)", fontsize=9)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("Cytology Attribution Overlay", color=status_col, fontweight='bold', fontsize=9)
        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.show()
    cam_engine.remove_hooks()


# =====================================================================
# 11. TEST & COMPREHENSIVE DIAGNOSTIC REPORTING
# =====================================================================
def test_and_report_advanced(model, test_loader, device, model_name="EfficientNetV2-S"):
    """Evaluates the model, printing clinical metrics and plotting Confusion Matrix and ROC-AUC."""
    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            logits, _ = model(inputs, None)
            probs = F.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_probs)

    print(f"\n{'='*20} {model_name.upper()} TEST METRICS {'='*20}")
    print(f"Accuracy: {acc*100:.2f}% | F1-Score: {f1:.4f} | ROC-AUC: {auc:.4f}")
    print(f"Sensitivity (Recall): {recall*100:.2f}% | Specificity: {specificity*100:.2f}% | Precision: {precision*100:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=["Healthy", "Blast"], yticklabels=["Healthy", "Blast"])
    axes[0].set_title(f"{model_name} - Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    axes[1].plot(fpr, tpr, label=f"AUC = {auc:.3f}", color='darkorange', lw=2)
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.6)
    axes[1].set_title(f"{model_name} - ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    return {"accuracy": acc, "precision": precision, "recall": recall, "specificity": specificity, "f1": f1, "auc": auc}
