import torch.utils.data as data
from PIL import Image
from torchvision import datasets, transforms
import os
import os.path
import nltk

""" 
    The following codes are copied and modified from PyTorch's COCO Caption dataset loader
    https://github.com/pytorch/vision/blob/master/torchvision/datasets/coco.py

    What's changed:
        COCO dataset provide 5 different captions for the same image. We only need 1 for training
"""


class CocoCaptionsTrain(data.Dataset):
    """`MS Coco Captions <http://mscoco.org/dataset/#captions-challenge2015>`_ Dataset.

    """

    def __init__(self, vocab, root, annFile, transform=None):
        from pycocotools.coco import COCO
        self.root = os.path.expanduser(root)
        self.coco = COCO(annFile)
        self.vocab = vocab
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of indices for the image.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        target = [ann['caption'] for ann in anns]

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # Convert caption to word ids
        target = target[0]
        tokens = nltk.tokenize.word_tokenize(str(target).lower)
        target = []
        target.append(self.vocab.word2vec("<start>"))
        target.extend([self.vocab.word2vec(token) for token in tokens])
        target.append(self.vocab.word2vec("<end>"))
        target = torch.Tensor(target)
        return img, target

    def __len__(self):
        return len(self.ids)


def train_collate_fn(data):
    """ Costomized mini-batch creation function 
    
    Args:
        data: Tuple(image, target) - the result from CocoCaptionsTrain

    Returns:
        tuple: Tuple (images, targets, lengths) - batch of images, batch of targets (with padding) and batch of valid lengths for these targets.

    """
    batch_size = len(data)
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # create mini-batch image
    images = torch.stack(images, 0)

    # create mini-batch target
    lengths = [len(caption) for caption in captions]
    minibatch_maxlen = max(lengths)
    targets = torch.zeros(batch_size, minibatch_maxlen).long()

    # copy the valid parts into targets
    for i, cap in enumerate(captions):
        targets[i, :lengths[i]] = cap[:lengths[i]]

    return images, targets, lengths


def get_train_loader(vocab, root, annFile, transform, batch_size, shuffle=True, num_workers=1):
    train_set = CocoCaptionsTrain(vocab, root, annFile, transform)
    train_set_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle,
                                                   num_workers=num_workers, collate_fn=train_collate_fn)
    return train_set_loader


def get_val_loader(root, annFile, transform, shuffle=False, num_workers=1):
    val_set = datasets.CocoCaptions(root, annFile, transform=transform, target_transform=None)
    val_set_loader = torch.utils.data.DataLoader(val_set, batch_Size=1, shuffle=shuffle, num_workers=num_workers)
    return val_set_loader
