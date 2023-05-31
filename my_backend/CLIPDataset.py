from torch.utils.data import Dataset
from PIL import Image
import clip
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CLIPDataset(Dataset):
    
    def __init__(self, image_paths, labels, preprocess):
        self.image_paths = image_paths
        self.labels = labels
        self.preprocess = preprocess

    def __getitem__(self, index):
        image = self.preprocess(Image.open(self.image_paths[index]))
        label = clip.tokenize(self.labels[index])[0]
        return image, label

    def __len__(self):
        return len(self.image_paths)