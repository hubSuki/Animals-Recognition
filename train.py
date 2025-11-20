# 断点续训 + 最终导出纯权重（兼容 UI）
import os, torch, torch.nn as nn, torch.optim as optim
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
from UI.model import AnimalCNN

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tf = T.Compose([
    T.RandomRotation(15), T.RandomHorizontalFlip(), T.ColorJitter(0.4, 0.4, 0.4, 0.1),
    T.RandomResizedCrop(148, scale=(0.8, 1.0)), T.ToTensor(), T.Normalize(mean, std), T.RandomErasing(p=0.5)
])
test_tf = T.Compose([T.Resize((148, 148)), T.ToTensor(), T.Normalize(mean, std)])

CKPT = './weights/checkpoint.pt'    # 断点字典
BEST = './weights/new_best_model.pth'   # UI 加载路径

def main():
    train_set = ImageFolder('./dataset/train', transform=train_tf)
    test_set  = ImageFolder('./dataset/test',  transform=test_tf)
    batch = 64  # 显存大小
    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AnimalCNN(num_classes=len(train_set.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    # lr：Adam 初始学习率；weight_decay：过拟合明显再加大 1e-2

    start_epoch = 1
    best_acc = 0.0
    if os.path.isfile(CKPT):
        print(f'>>>> 加载断点 {CKPT}')
        ckpt = torch.load(CKPT, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt['best_acc']

    epochs = 25
    for ep in range(start_epoch, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, ncols=80, desc=f'Epoch {ep}/{epochs}')
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
            pbar.set_postfix(loss=running_loss/total, acc=correct/total)

        model.eval()
        with torch.no_grad():
            test_correct, test_total = 0, 0
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                test_correct += (model(x).argmax(1) == y).sum().item()
                test_total += y.size(0)
        test_acc = test_correct / test_total
        best_acc = max(best_acc, test_acc)
        print(f'>>> Test Acc: {test_acc:.4f}   best: {best_acc:.4f}')

        # 保存断点（字典形式）
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': ep,
                    'best_acc': best_acc}, CKPT)

    # 训练完成：导出纯权重给 UI 使用
    torch.save(model.state_dict(), BEST)
    os.remove(CKPT)          # 清理断点
    print(f'train done → {BEST}')

if __name__ == '__main__':
    main()