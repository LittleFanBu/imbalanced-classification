from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob


class my_dataset(Dataset):
    def __init__(self, store_path, split, data_transform=None):
        self.store_path = store_path
        self.split = split
        self.name = {'with_mask': 0, 'without_mask': 1}
        self.transforms = data_transform
        self.img_list = []  # store the path of img
        self.label_list = []  # store the class label
        for file in glob.glob(self.store_path + '/' + split + '/*png'):
            cur_path = file.replace('\\', '/')  # the path of each img
            self.img_list.append(cur_path)
            if split == 'with_mask':
                self.label_list.append(0)  # mark "with mask" as 0
            elif split == 'without_mask':
                self.label_list.append(1)  # mark "without mask" as 1

    def __getitem__(self, item):
        img = Image.open(self.img_list[item]).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        label = self.label_list[item]
        return img, label

    def __len__(self):
        return len(self.img_list)
