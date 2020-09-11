import torch
from torch.autograd import Variable
from torchvision import transforms
from pathlib import Path
import pandas as pd
from PIL import Image
import tkinter as tk
from tkinter import filedialog


DATASET_ROOT = ''
PATH_TO_WEIGHTS = ''


def test():
    data_transform = transforms.Compose([
        transforms.Scale(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    root = tk.Tk()
    root.withdraw()
    DATASET_ROOT = filedialog.askdirectory()
    PATH_TO_WEIGHTS = filedialog.askopenfilename()
    
    dataset_root = Path(DATASET_ROOT)
    classes = [_dir.name for _dir in dataset_root.joinpath('train').glob('*')]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(PATH_TO_WEIGHTS)
    model = model.to(device)
    model.eval()

    sample_submission = pd.read_csv(str(dataset_root.joinpath('sample_submission.csv')))
    submission = sample_submission.copy()
    for i, filename in enumerate(sample_submission['file']):
        image = Image.open(str(dataset_root.joinpath('test').joinpath(filename))).convert('RGB')
        image = data_transform(image).unsqueeze(0)
        inputs = Variable(image.to(device))
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        submission['species'][i] = classes[preds[0]]

    submission.to_csv(str(dataset_root.joinpath('submission.csv')), index=False)


if __name__ == '__main__':
    test()
