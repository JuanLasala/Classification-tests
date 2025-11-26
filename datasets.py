import os
from PIL import Image
from torch.utils.data import Dataset

class ImageClassificationDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths      # lista de paths completos
        self.transform = transform

        # Crear un diccionario clase → número
        self.classes = sorted(list({os.path.basename(os.path.dirname(p)) for p in image_paths}))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_name = os.path.basename(os.path.dirname(img_path))
        label = self.class_to_idx[label_name]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label
