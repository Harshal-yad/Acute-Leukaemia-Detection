import os
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)



# REPRODUCIBILITY & SEED
def set_seed(seed=42):
    """Sets random seeds for reproducibility across Python, NumPy, and PyTorch."""
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# 2. DATASET INGESTION & PARSING
def load_and_combine_datasets(idb1_dir="ALL_IDB1/im", idb2_png_dir="ALL_IDB2/img_png"):
    """
    Parses and unifies ALL_IDB1 and ALL_IDB2 image directories into a single DataFrame.
    Labels: 0 = Non-Blast (Healthy), 1 = Blast (ALL)
    """
    
    data = []
    # Process ALL_IDB1
    for f in os.listdir(idb1_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            match = re.search(r'_([01])\.', f)
            if match:
                label = int(match.group(1))
                data.append({
                    "filepath": os.path.join(idb1_dir, f),
                    "filename": f,
                    "dataset": "ALL_IDB1",
                    "label": label,
                    "class_name": "Blast (ALL)" if label == 1 else "Non-Blast (Healthy)"
                })
                
    # Process ALL_IDB2
    target_idb2 = idb2_png_dir
    for f in os.listdir(target_idb2):
        if f.lower().endswith(('.png', '.jpg')):
            match = re.search(r'_([01])\.', f)
            if match:
                label = int(match.group(1))
                data.append({
                    "filepath": os.path.join(target_idb2, f),
                    "filename": f,
                    "dataset": "ALL_IDB2",
                    "label": label,
                    "class_name": "Blast (ALL)" if label == 1 else "Non-Blast (Healthy)"
                })

                
    df = pd.DataFrame(data)
    print(f"[Dataset] Loaded {len(df)} images total.")
    
    return df



# 3. PYTORCH DATASET Class
class LeukemiaDataset(Dataset):
    """Custom PyTorch Dataset for Leukemia microscopic images."""
    
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        label = row["label"]
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

# 3.1 DATA Augmentation
def get_transforms(img_size=224):
    """Returns training transforms (with augmentation) and validation/test transforms."""
    
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


# 3.2 Dataloader (Train / Val / Test)
def create_dataloaders(df, batch_size=16, img_size=224, test_size=0.30, val_ratio_of_test=0.50):
    """
    Performs stratified train/val/test split and returns PyTorch DataLoaders.
    Note: drop_last=True is set on train_loader to prevent single-sample batch normalization errors.
    """
    
    train_df, temp_df = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df["label"]
    )
    
    val_df, test_df = train_test_split(
        temp_df, test_size=val_ratio_of_test, random_state=42, stratify=temp_df["label"]
    )
    
    train_tf, eval_tf = get_transforms(img_size)
    train_ds = LeukemiaDataset(train_df, transform=train_tf)
    
    val_ds = LeukemiaDataset(val_df, transform=eval_tf)
    
    test_ds = LeukemiaDataset(test_df, transform=eval_tf)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"[Splits] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    return train_loader, val_loader, test_loader, test_ds


# 3.3 De-Normalize
def denormalize(tensor):
    """Denormalizes ImageNet standardized tensor back to [0, 1] range for visualization."""
    inv_mean = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]
    inv_std = [1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225]
    inv_norm = transforms.Normalize(mean=inv_mean, std=inv_std)
    
    return inv_norm(tensor).clamp(0, 1)



# 4. SINGLE MODEL BUILDER
def build_classifier_head(in_features, num_classes=2, hidden_dim=256, dropout_p=0.3):
    """Custom multi-layer classification head replacing the original top layers."""
    
    return nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_p),
        nn.Linear(hidden_dim, num_classes)
    )
    
def get_single_model(model_name="resnet50", num_classes=2, freeze_backbone=True):
    """
    Builds a single pretrained model, freezes backbone if requested, 
    replaces classification head, and identifies the target convolutional layer for Grad-CAM.
    
    Supported: 'resnet50', 'vgg16_bn', 'densenet121', 'efficientnet_b0'
    """
    
    model_name = model_name.lower()
    
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = build_classifier_head(in_features, num_classes)
        target_layer = model.layer4[-1]
        
    elif model_name == "vgg16_bn":
        model = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.classifier[0].in_features
        model.classifier = build_classifier_head(in_features, num_classes)
        target_layer = model.features[-3]
        
    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.classifier.in_features
        model.classifier = build_classifier_head(in_features, num_classes)
        target_layer = model.features.norm5
        
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )
        target_layer = model.features[-1]
    else:
        raise ValueError(f"Unsupported model: {model_name}")
        
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"[{model_name.upper()}] Total: {total_p:,} | Trainable: {train_p:,} ({train_p/total_p*100:.2f}%)")
    
    return model, target_layer



# 5. SINGLE MODEL TRAINING
def train_single_epoch(model, dataloader, criterion, optimizer, device):
    """Runs a single training epoch."""
    
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        
    return running_loss / total, correct / total


def train_model(model, train_loader, val_loader, device, epochs=20, lr=1e-3, 
                model_name="model", resume=True):
    """
    Full training pipeline for a single model:
    - If resume=True and 'checkpoint_<model_name>.pth' exists, resumes from the last finished epoch.
    - Saves 'checkpoint_<model_name>.pth' after every epoch for safe resuming.
    - Saves 'best_<model_name>.pth' whenever validation accuracy achieves a new best.
    """
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    checkpoint_path = f"checkpoint_{model_name}.pth"
    best_path = f"best_{model_name}.pth"

    start_epoch = 1
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Resume from checkpoint if it exists
    if resume and os.path.exists(checkpoint_path):
        start_epoch, best_val_acc, history = load_checkpoint(
            model, optimizer, scheduler, checkpoint_path, device
        )
        
        if start_epoch > epochs:
            print(f"[Notice] Model has already been trained for {start_epoch - 1}/{epochs} epochs. Set a higher epoch count to continue.")
            # Load best weights
            if os.path.exists(best_path):
                model.load_state_dict(torch.load(best_path, map_location=device))
                
            return model, history

    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_acc = train_single_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _, _ = evaluate_single_model(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 1. Update best model checkpoint
        if epoch == 1 or val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

        # 2. Save full resumable checkpoint at every epoch
        save_checkpoint(model, optimizer, scheduler, epoch, best_val_acc, history, checkpoint_path)

        if epoch % 1 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} (Best: {best_val_acc:.4f})")

    # Load best weights for downstream evaluation
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    return model, history

    

# 6. CHECKPOINTING & RESUME CAPABILITIES
def save_checkpoint(model, optimizer, scheduler, epoch, best_val_acc, history, checkpoint_path):
    """Saves full training state for seamless resuming later."""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_acc': best_val_acc,
        'history': history
    }
    torch.save(state, checkpoint_path)


    
def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """Loads saved checkpoint state into model, optimizer, and scheduler."""
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_acc = checkpoint['best_val_acc']
    history = checkpoint.get('history', {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []})
    
    print(f"[Checkpoint] Resuming from Epoch {start_epoch} (Prior Best Val Acc: {best_val_acc*100:.2f}%)")
    
    return start_epoch, best_val_acc, history


def load_trained_model(model_name, weights_path, device, num_classes=2):
    """
    Directly builds the model and loads weights for fast evaluation or Grad-CAM
    without needing to run the training loop at all.
    """
    
    model, target_layer = get_single_model(model_name, num_classes=num_classes, freeze_backbone=True)
    state = torch.load(weights_path, map_location=device)
    
    # Handles both full checkpoints and raw state_dicts
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
        
    model = model.to(device)
    model.eval()
    
    print(f"[Loaded] Successfully loaded weights from {weights_path} for {model_name.upper()}")
    
    return model, target_layer




# 7. EVALUATION
def evaluate_single_model(model, dataloader, criterion, device):
    """Evaluates a single model on a dataloader (validation or test set)."""
    
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets, all_probs = [], [], []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            
    return running_loss / total, correct / total, np.array(all_targets), np.array(all_preds), np.array(all_probs)


def test_and_report(model, test_loader, device, model_name="Model"):
    """Computes comprehensive test metrics, prints classification stats, and plots confusion matrix and ROC curve."""
    
    criterion = nn.CrossEntropyLoss()
    loss, acc, y_true, y_pred, y_probs = evaluate_single_model(model, test_loader, criterion, device)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_probs)
    
    print(f"\n{'='*20} {model_name.upper()} TEST RESULTS {'='*20}")
    print(f"Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")
    print(f"Sensitivity (Recall): {recall*100:.2f}% | Specificity: {specificity*100:.2f}%")
    print(f"Precision: {precision*100:.2f}% | F1-Score: {f1:.4f} | ROC-AUC: {auc:.4f}")
    
    # Plot Confusion Matrix and ROC Curve
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=["Healthy", "Blast"], yticklabels=["Healthy", "Blast"])
    axes[0].set_title(f"{model_name.upper()} - Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    axes[1].plot(fpr, tpr, label=f"AUC = {auc:.3f}", color='darkorange', lw=2)
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.6)
    axes[1].set_title(f"{model_name.upper()} - ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "auc": auc}




# GradCam
class GradCAM:
    """Pure PyTorch Grad-CAM module for visualizing convolutional feature attribution."""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        # Ensure parameters require gradients so backward hooks capture activations
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
        output = self.model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
        score = output[:, target_class]
        self.model.zero_grad()
        score.backward(retain_graph=True)
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = nn.functional.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().detach().numpy()
        cam_min, cam_max = np.min(cam), np.max(cam)
        cam = (cam - cam_min) / (cam_max - cam_min) if cam_max > cam_min else np.zeros_like(cam)
        confidence = torch.softmax(output, dim=1)[0, target_class].item()
        
        return cam, target_class, confidence
        
    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def overlay_gradcam(original_img_np, heatmap, alpha=0.45):
    """Overlays Grad-CAM heatmap over original RGB image."""
    
    cmap = plt.get_cmap('jet')
    heatmap_colored = cmap(heatmap)[:, :, :3]
    overlay = (1.0 - alpha) * original_img_np + alpha * heatmap_colored
    
    return np.clip(overlay, 0, 1), heatmap_colored


def visualize_gradcam_samples(model, target_layer, test_dataset, device, num_samples=4):
    """Visualizes Grad-CAM on representative test images (True Positives and True Negatives)."""
    
    grad_cam = GradCAM(model, target_layer)
    indices = []
    
    for idx in range(len(test_dataset)):
        _, label = test_dataset[idx]
        if label.item() == 1 and len([i for i in indices if test_dataset[i][1].item() == 1]) < num_samples // 2:
            indices.append(idx)
        elif label.item() == 0 and len([i for i in indices if test_dataset[i][1].item() == 0]) < num_samples // 2:
            indices.append(idx)
        if len(indices) == num_samples:
            break
            
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 3 * num_samples))
    
    for row, sample_idx in enumerate(indices):
        img_tensor, true_label = test_dataset[sample_idx]
        input_batch = img_tensor.unsqueeze(0).to(device)
        heatmap, pred_class, confidence = grad_cam.generate_heatmap(input_batch)
        orig_np = denormalize(img_tensor).permute(1, 2, 0).numpy()
        overlay, _ = overlay_gradcam(orig_np, heatmap)
        true_txt = "Blast (1)" if true_label.item() == 1 else "Healthy (0)"
        pred_txt = "Blast (1)" if pred_class == 1 else "Healthy (0)"
        status_col = "green" if pred_class == true_label.item() else "red"
        axes[row, 0].imshow(orig_np)
        axes[row, 0].set_title(f"Original | True: {true_txt}", fontsize=9)
        axes[row, 0].axis("off")
        axes[row, 1].imshow(heatmap, cmap="jet")
        axes[row, 1].set_title(f"Heatmap | Pred: {pred_txt} ({confidence*100:.1f}%)", fontsize=9)
        axes[row, 1].axis("off")
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("Overlay", color=status_col, fontweight='bold', fontsize=9)
        axes[row, 2].axis("off")
        
    plt.tight_layout()
    plt.show()
    grad_cam.remove_hooks()

