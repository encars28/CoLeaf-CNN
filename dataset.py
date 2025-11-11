import os
from PIL import Image
from torch.utils.data import Dataset
from tqdm.notebook import tqdm

class CoLeafDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(data_dir)])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for class_name in tqdm(self.classes, desc="Loading classes"):
            class_path = os.path.join(self.data_dir, class_name)
            for filename in tqdm(os.listdir(class_path), desc=f"Loading images for class {class_name}", leave=False):
                image = Image.open(os.path.join(class_path, filename)).convert('RGB')
                # Aplicar las transformaciones
                if self.transform:
                    image = self.transform(image)
                
                self.images.append(image)
                self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
    

# Create and save dataset
if __name__ == "__main__":
    import os
    import torch
    from torchvision import transforms
    
    DATASET_OUTPUT_DIR = './output/datasets/dataset.pt'
    DATASET_PATH = './CoLeaf DATASET'
    IMG_SIZE = 224
    
    
    dataset = CoLeafDataset(
        data_dir=os.path.join(DATASET_PATH),
        transform=transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),          
            transforms.ToTensor(),           
            transforms.Normalize(            
                mean=[0.5, 0.5, 0.5], 
                std=[0.5, 0.5, 0.5]
            )
        ])
    )

    torch.save(dataset, DATASET_OUTPUT_DIR)
