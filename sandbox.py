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
from skimage.color import label2rgb
from tqdm import tqdm
from matplotlib.patches import Circle

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


def plot_line(ax, center, slope, length=1):
    if slope > 30:
        slope = 30
    elif slope < -30:
        slope = -30
    else:
        slope = slope

    if -0.01 < slope < 0.01:
        length = 2
    elif (slope > 5) or (slope < -5):
        length = 0.2
    else:
        length=length



    b = center[1] - slope * center[0]
    pt1 = (center[0] - length, slope * (center[0] - length) + b)
    pt2 = (center[0] + length, slope * (center[0] + length) + b)

    ax.plot((pt1[0], center[0]), (pt1[1], center[1]), color='red', linewidth=0.5)
    ax.plot((center[0], pt2[0]), (center[1], pt2[1]), color='red', linewidth=0.5)





class SuperpixelDataset(torch.utils.data.Dataset):
    """Invariant dataset."""
    def __init__(self,orig_dataset, num_edges):
        """
        Args:
            orig_dataset (Dataset): dataset
        """
        self.orig_dataset = orig_dataset
        self.num_edges = num_edges
    def __len__(self):
        return len(self.orig_dataset)

    def __getitem__(self, idx):
        img,target = self.orig_dataset[idx]
        img = np.float32(np.asarray(img[0,:,:]))/255
        labels = slic(img, n_segments=25, compactness=0.5, sigma=0.1, start_label=0)
        p = regionprops(labels+1,intensity_image=img)
        g = graph.rag_mean_color(img, labels)
        feats = []
        coords = []
        for node in g.nodes:
            color = p[node]['mean_intensity']
            orientation = p[node].orientation#['orientation']
            # print(orientation)
            invariants = p[node]['moments_hu']
            center = torch.Tensor(p[node]['centroid']).unsqueeze(0)
            feat = torch.cat([torch.Tensor([color]),torch.Tensor(invariants)]).unsqueeze(0)
            feats.append(feat)
            coords.append(center)


        locations = []
        orientations = []


        for edge in list(g.edges)[:self.num_edges]: # 'cropping'
            node1 = edge[0]
            node2 = edge[1]

            node1_loc = p[node1].centroid
            node2_loc = p[node2].centroid

            edge_loc = (np.array(node1_loc) + np.array(node2_loc))/2
            locations.append(torch.Tensor(np.array([edge_loc])))

            orientation = np.rad2deg(np.arctan2(node2_loc[-1] - node2_loc[0], node1_loc[-1] - node1_loc[0]))
            orientations.append(torch.Tensor(np.array([orientation])))

        # 'padding'
        if len(g.edges) <= self.num_edges:
            for i in range(self.num_edges - len(g.edges)):
                locations.append(locations[i])
                orientations.append(orientations[i])


        locations = torch.cat(locations , dim=0)
        orientations = torch.cat(orientations, dim=0)


        feats = torch.cat(feats,dim=0)
        coords = torch.cat(coords,dim=0)

        return dict(img=img, locations=locations, orientations=orientations, target=target, labels=labels)




total_transform = transforms.Compose([
                #RandomReduceScale(0.3,1),
                #RandomRotate(-180,180),
                transforms.ToTensor()])
total_transform_test = transforms.Compose([
                #RandomReduceScale(0.3,1),
                #RandomRotate(-180,180),
                transforms.ToTensor()])




mnist_train = SuperpixelDataset(datasets.MNIST('data', train=True, download=True,transform=total_transform), num_edges=70)
mnist_test = SuperpixelDataset(datasets.MNIST('data', train=False, download=True,transform=total_transform_test), num_edges=70)





trainloader = torch.utils.data.DataLoader(mnist_train, batch_size=10, shuffle=True)


sample = next(iter(trainloader))
fig, ax = plt.subplots(2)

ax[0].imshow(sample['img'][0], cmap='Blues_r')

for i in range(sample['locations'][0].size(0)):
    loc = sample['locations'][0][i]
    orientation = sample['orientations'][0][i]
    circ = Circle(loc, 0.3, color='red')
    slope = np.tan(orientation.numpy())

    plot_line(ax[0], loc, slope)
    ax[0].add_patch(circ)




# segments = slic(sample['img'][0], n_segments=25, compactness=0.5, sigma=0.1, start_label=1)
# ax[0].imshow(mark_boundaries(sample['img'][0], sample['labels'][0].numpy()))
# ax[0].imshow(mark_boundaries(sample['img'][0], segments))
plt.show()
