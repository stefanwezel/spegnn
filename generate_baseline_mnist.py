import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries, find_boundaries
from skimage.util import img_as_float
from skimage.future import graph
from skimage.measure import regionprops
from skimage.color import label2rgb
from skimage import io
from scipy.spatial import distance
import os
import json
import pickle

from utils import get_bearing, get_highest_contrast_neighbor, get_neighbors, plot_line


from models import E_GCL, EGNN, get_edges_batch






def collate(batch):
    batch_feats = [b[0] for b in batch]
    batch_coords = [b[1] for b in batch]
    batch_targets = [b[2] for b in batch]
    batch_imgs = [b[3] for b in batch]
    batch_segments = [b[4] for b in batch]
    batch_feats = torch.nn.utils.rnn.pad_sequence(batch_feats,batch_first=True)
    batch_coords = torch.nn.utils.rnn.pad_sequence(batch_coords,batch_first=True)
    n_nodes = batch_feats.shape[1]
    batch_size = batch_feats.shape[0]
    edges, edge_attr = get_edges_batch(n_nodes, batch_size)
    batch_coords = batch_coords.view(-1,2) / 28
    batch_feats = batch_feats.view(batch_coords.shape[0],-1)
    batch_targets = torch.LongTensor(batch_targets)
    
    return dict(
                feats=batch_feats,
                coords=batch_coords,
                targets=batch_targets,
                edges=edges,
                n_nodes=n_nodes,
                batch_size=batch_size,
                # for plotting
                imgs = batch_imgs,
                segments = batch_segments,
        )
    # return [batch_feats,batch_coords,batch_targets,edges,n_nodes,batch_size]


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

        img_rgb = np.zeros((28,28,3))
        img_rgb[:,:,2] = img
        img_rgb[:,:,1] = img
        img_rgb[:,:,0] = img
        # img_rgb *= 255

        # labels = slic(img_rgb, n_segments=25, compactness=0.5, sigma=0.1)
        labels = slic(img_rgb*255, n_segments = 25, compactness = 50, sigma = 0.01, start_label=0)

        # labels = slic(img_rgb*255,
        #                     n_segments = 50,
        #                     compactness = 50,
        #                     sigma = 0.01,
        #                     # start_label=0,
        #                     )

        p = regionprops(1+labels,intensity_image=img)
        g = graph.rag_mean_color(img_rgb, labels)
        feats = []
        coords = []
        for node in g.nodes:
            color = p[node]['mean_intensity']
            invariants = p[node]['moments_hu']
            center = torch.Tensor(p[node]['centroid']).unsqueeze(0)
            # print([f"{invariant:.2f}" for invariant in invariants])
            feat = torch.cat([torch.Tensor([color]),torch.Tensor(invariants)]).unsqueeze(0)
            feats.append(feat)
            coords.append(center)
        feats = torch.cat(feats,dim=0)
        coords = torch.cat(coords,dim=0)
        return (feats,coords,target, img,labels)






total_transform = transforms.Compose([
                transforms.ToTensor()])
total_transform_test = transforms.Compose([
                transforms.ToTensor()])


mnist_train = SuperpixelDataset(datasets.MNIST('../data', train=True, download=True,transform=total_transform))
mnist_test = SuperpixelDataset(datasets.MNIST('../data', train=False, download=True,transform=total_transform_test))



# kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}

train_loader = torch.utils.data.DataLoader(mnist_train,
    batch_size=1, shuffle=True, collate_fn=collate)
test_loader = torch.utils.data.DataLoader(mnist_test,
    batch_size=1, shuffle=True,  collate_fn=collate)


sample = next(iter(train_loader))
# print(sample.keys())




feats = []
coords = []
targets = []
is_trains = []


# import matplotlib.pyplot as plt
# plt.imshow(mark_boundaries(sample['imgs'][0], sample['segments'][0]))

# plt.scatter(sample['coords'][:,1]*28, sample['coords'][:,0]*28)
# plt.show()



for i, sample in enumerate(train_loader):
    if i < 6000:
        # img_name = f"./graph_mnist/train_imgs/mnist_{i:05d}.npy"
        lens = [len(sample[key]) for key in ['feats', 'coords']]
        assert len(set(lens)) == 1 # check whether all elements have same lengths

        feats.append(sample['feats'])
        coords.append(sample['coords'])

        targets.append(sample['targets'])
        is_trains.append(1)
    else:
        break


data = dict(
    feats=feats,
    coords=coords,
    targets=targets,
    is_trains=is_trains,
    )


with open("./graph_mnist/baseline_train_data.pickle", "wb") as fp:
    pickle.dump(data, fp)






for i, sample in enumerate(test_loader):
    if i < 1000:
        # img_name = f"./graph_mnist/train_imgs/mnist_{i:05d}.npy"
        lens = [len(sample[key]) for key in ['feats', 'coords']]
        assert len(set(lens)) == 1 # check whether all elements have same lengths

        feats.append(sample['feats'])
        coords.append(sample['coords'])

        targets.append(sample['targets'])
        is_trains.append(0)
    else:
        break


data = dict(
    feats=feats,
    coords=coords,
    targets=targets,
    is_trains=is_trains
    )


with open("./graph_mnist/baseline_test_data.pickle", "wb") as fp:
    pickle.dump(data, fp)




