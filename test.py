# （兼容训练阶段结构 + 安全加载）
import os, torch, torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- 1. 模型结构：与训练阶段完全一致 ----------
class AnimalCNN(nn.Module):
    def __init__(self, num_classes: int, drop: float = 0.4):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(drop),
            nn.Linear(128 * 9 * 9, 256),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.feat(x))

# ---------- 2. 数据 ----------
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
test_tf = transforms.Compose([
    transforms.Resize((148, 148)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
test_set = datasets.ImageFolder('./dataset/test', transform=test_tf)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)
animal_labels = test_set.classes
num_classes = len(animal_labels)

# ---------- 3. 加载模型（兼容纯权重 & 断点字典） ----------
model = AnimalCNN(num_classes).to(device)
ckpt = torch.load('./weights/new_best_model.pth', map_location=device, weights_only=False)
# 如果是断点字典，提取模型权重；如果是纯权重，直接加载
model_weights = ckpt.get('model', ckpt)
model.load_state_dict(model_weights, strict=True)
model.eval()

criterion = nn.CrossEntropyLoss()

# ---------- 4. 评估 ----------
def run_eval(loader):
    loss_tot, correct, total = 0., 0, 0
    all_y, all_y_hat = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc='eval'):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_tot += criterion(out, y).item()
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            all_y.extend(y.cpu().numpy())
            all_y_hat.extend(pred.cpu().numpy())
    acc = correct / total
    return loss_tot / len(loader), acc, all_y, all_y_hat

test_loss, test_acc, y_true, y_pred = run_eval(test_loader)

# ---------- 5. 指标报告 ----------
report = classification_report(y_true, y_pred, target_names=animal_labels, digits=4)
print('Test Accuracy: {:.4f}   Loss: {:.4f}'.format(test_acc, test_loss))
print('Classification Report:\n', report)

os.makedirs('./training_results', exist_ok=True)
with open('./training_results/classification_report.txt', 'w') as f:
    f.write('Test Classification Report:\n')
    f.write(report)

# ---------- 6. F1 可视化 ----------
f1s = f1_score(y_true, y_pred, average=None)
plt.figure(figsize=(6, 4))
sns.barplot(x=animal_labels, y=f1s, hue=animal_labels, palette='Blues_d', legend=False)
for i, v in enumerate(f1s):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
plt.ylim(0, 1.05)
plt.title('F1-Score per class')
plt.tight_layout()
plt.savefig('./Image/F1_bar.png', dpi=200)
plt.show()

cm = confusion_matrix(y_true, y_pred)
f1_mat = np.zeros_like(cm, dtype=float)
for i in range(num_classes):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    f1_mat[i, i] = 2 * tp / (2 * tp + fp + fn + 1e-8)

plt.figure(figsize=(5, 4))
sns.heatmap(f1_mat, annot=True, fmt='.3f', cmap='Greens',
            xticklabels=animal_labels, yticklabels=animal_labels)
plt.title('F1-Score Matrix (diagonal)')
plt.tight_layout()
plt.savefig('./Image/F1_heatmap.png', dpi=200)
plt.show()

# ---------- 7. 预测示例 & 混淆矩阵 ----------
def display_predictions(loader, num_img=10):
    model.eval()
    imgs, lbls, preds = [], [], []
    with torch.no_grad():
        for x, y in loader:
            if len(imgs) >= num_img:
                break
            x, y = x.to(device), y.to(device)
            out = model(x)
            pr = out.argmax(1)
            imgs.extend(x.cpu())
            lbls.extend(y.cpu())
            preds.extend(pr.cpu())
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    for i in range(num_img):
        img = imgs[i].permute(1, 2, 0).numpy() * std + mean
        img = np.clip(img, 0, 1)
        true = animal_labels[lbls[i]]
        pred = animal_labels[preds[i]]
        color = 'green' if lbls[i] == preds[i] else 'red'
        axes[i].imshow(img)
        axes[i].set_title(f'T:{true}\nP:{pred}', color=color, fontsize=9)
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig('./Image/test_predictions.png', dpi=200)
    plt.show()

def plot_cm():
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=animal_labels, yticklabels=animal_labels)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('./Image/test_confusion_matrix.png', dpi=200)
    plt.show()

display_predictions(test_loader)
plot_cm()