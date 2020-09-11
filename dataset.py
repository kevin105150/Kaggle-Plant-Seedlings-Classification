from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


class PlantSeedlingDataset(Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.root = Path(root)
        self.img = []
        self.classes = []
        self.num_classes = 0
        

        if self.root.name == 'train':
            for i, class_dir in enumerate(self.root.glob('*')):
                for file in class_dir.glob('*'):
                    self.img.append([i,file])
                self.classes.append(class_dir)
        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        image = Image.open(self.img[index][1]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.img[index][0]