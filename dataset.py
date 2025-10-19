import os
from PIL import Image
from torch.utils.data import Dataset

class CoLeafDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(data_dir)])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        for class_name in self.classes:
            class_path = os.path.join(self.data_dir, class_name)
            for filename in os.listdir(class_path):
                self.image_paths.append(os.path.join(class_path, filename))
                self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Cargar la imagen y convertirla a RGB (por si acaso)
        image = Image.open(img_path).convert('RGB')
        
        # Aplicar las transformaciones
        if self.transform:
            image = self.transform(image)

        return image, label

