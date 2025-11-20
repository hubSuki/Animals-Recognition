NUM_CLASSES = 10                            # 类别数 → 必须与 dataset 文件夹子目录个数一致
MODEL_PATH  = './weights/new_best_model.pth'    # UI 加载的权重
ICON_PATH   = './UI/icons/动物识别.png'
SAMPLE_EVERY = 8                            # 视频隔多少帧采样一帧（越大越快越粗糙）

ANIMAL_LABELS = [
    '🦋 蝴蝶', '🐱 猫咪', '🐔 小鸡', '🐄 奶牛', '🐕 小狗',
    '🐘 大象', '🐎 马儿', '🐑 绵羊', '🕷️ 蜘蛛', '🐿️ 松鼠'
]                                           # 顺序必须与 ImageFolder 的 class_to_idx 一致

from torchvision import transforms
# 统一均值方差
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(148, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
    transforms.RandomErasing(p=0.3)
])                                         # 数据增强强度

TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize((148, 148)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

TRANSFORMS = TEST_TRANSFORMS