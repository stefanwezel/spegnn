import numpy as np
import torch
from torchvision import datasets, transforms
from types import SimpleNamespace

import matplotlib.pyplot as plt
from skimage.data import astronaut
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries, find_boundaries
from skimage.util import img_as_float
from skimage.future import graph
from skimage.measure import regionprops
from tqdm import tqdm

# from egnn_clean import E_GCL, EGNN, get_edges_batch


def collate(batch):
    batch_feats = [b[0] for b in batch]
    batch_coords = [b[1] for b in batch]
    batch_targets = [b[2] for b in batch]
    batch_feats = torch.nn.utils.rnn.pad_sequence(batch_feats,batch_first=True)
    batch_coords = torch.nn.utils.rnn.pad_sequence(batch_coords,batch_first=True)
    n_nodes = batch_feats.shape[1]
    batch_size = batch_feats.shape[0]
    edges, edge_attr = get_edges_batch(n_nodes, batch_size)
    batch_coords = batch_coords.view(-1,2) / 28
    batch_feats = batch_feats.view(batch_coords.shape[0],-1)
    batch_targets = torch.LongTensor(batch_targets)
    
    return [batch_feats,batch_coords,batch_targets,edges,n_nodes,batch_size]





class SuperpixelDataset(torch.utils.data.Dataset):
    """Whale dataset."""
    def __init__(self,orig_dataset):
        """
        Args:
            orig_dataset (Dataset): dataset
        """
        self.orig_dataset = orig_dataset
    def __len__(self):
        return len(self.orig_dataset)

    def __getitem__(self, idx):
        img,target = self.orig_dataset[idx]
        img = np.float32(np.asarray(img[0,:,:]))/255
        labels = slic(img, n_segments=25, compactness=0.5, sigma=0.1)
        p = regionprops(labels+1,intensity_image=img)
        g = graph.rag_mean_color(img, labels)
        feats = []
        coords = []
        for node in g.nodes:
            color = p[node]['mean_intensity']
            invariants = p[node]['moments_hu']
            center = torch.Tensor(p[node]['centroid']).unsqueeze(0)
            feat = torch.cat([torch.Tensor([color]),torch.Tensor(invariants)]).unsqueeze(0)
            feats.append(feat)
            coords.append(center)
        feats = torch.cat(feats,dim=0)
        coords = torch.cat(coords,dim=0)
        return (feats,coords,target)




total_transform = transforms.Compose([
                #RandomReduceScale(0.3,1),
                #RandomRotate(-180,180),
                transforms.ToTensor()])
total_transform_test = transforms.Compose([
                #RandomReduceScale(0.3,1),
                #RandomRotate(-180,180),
                transforms.ToTensor()])




mnist_train = SuperpixelDataset(datasets.MNIST('data', train=True, download=True,transform=total_transform))
mnist_test = SuperpixelDataset(datasets.MNIST('data', train=False, download=True,transform=total_transform_test))

print(len(mnist_test))