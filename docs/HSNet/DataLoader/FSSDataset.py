from docs.HSNet.DataLoader.CustomLoader import DatasetCustom
from docs.HSNet.DataLoader.PASCALLoader import DatasetPASCAL

from torch.utils.data import DataLoader
from torchvision import transforms

class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize):

        cls.datasets = {
            "custom": DatasetCustom,
            "pascal": DatasetPASCAL
        }

        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]
        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize

        cls.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(cls.img_mean, cls.img_std)])

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1, neg_inst_rate = True):
        # dataset = cls.datasets[benchmark](cls.datapath, transform=cls.transform, shot=shot, use_original_imgsize=cls.use_original_imgsize, experiment = experiment)
        # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility

        shuffle = split == 'trn'
        nworker = nworker if split == 'trn' else 0

        dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=cls.transform, split=split, shot=shot, use_original_imgsize=cls.use_original_imgsize, neg_inst_rate = neg_inst_rate)
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker)

        return dataloader