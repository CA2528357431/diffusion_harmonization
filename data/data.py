from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image


class DataSet(Dataset):

    def __init__(self, path=""):
        # 保存数据，ori保存未和谐化的图，tar保存和谐化的图
        self.ori_container = []
        self.tar_container = []

        # 数据长度，有多少(ori, tar)对
        self.length = 0

        # open图片文件后的transform
        # 这只是一个示例
        # resize为512*512
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.Resize(size=(512, 512), interpolation=Image.BICUBIC)
        ])

    def __getitem__(self, index):
        # 返回的格式----未和谐化的图，和谐化的图

        return (
            self.ori_container[index],
            self.tar_container[index]
        )

    def __len__(self):
        return self.length
