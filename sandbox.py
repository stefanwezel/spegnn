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
from skimage import io
from scipy.spatial import distance

from tqdm import tqdm
from matplotlib.patches import Circle
from PIL import Image
# from egnn_clean import E_GCL, EGNN, get_edges_batch




def collate_edges(batch):
    """ TODO """
    batch_imgs = [b['img_rgb'] for b in batch]
    batch_locations = [b['locations'] for b in batch]
    batch_orientations = [b['orientations'] for b in batch]
    batch_targets = [b['target'] for b in batch]
    batch_segments = [b['segments'] for b in batch]
    batch_locations = torch.nn.utils.rnn.pad_sequence(batch_locations,batch_first=True)
    batch_orientations = torch.nn.utils.rnn.pad_sequence(batch_orientations,batch_first=True)
    # print(batch_orientations.size())
    # n_nodes = batch_locations.shape[1]
    # batch_size = batch_locations.shape[0]
    # edges, edge_attr = get_edges_batch(n_nodes, batch_size)
    # batch_orientations = batch_orientations.view(-1,2) / 28
    # batch_locations = batch_locations.view(batch_orientations.shape[0],-1)
    # batch_targets = torch.LongTensor(batch_targets)
    
    # return [batch_locations,batch_orientations,batch_targets,edges,n_nodes,batch_size]
    return dict(
                img=batch_imgs,
                locations=batch_locations,
                orientations=batch_orientations,
                target=batch_targets,
                segments=batch_segments,
        )





def plot_line(ax, center, slope, length=1):
    """ TODO """
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




def get_neighbors(node, edges):
    """ TODO """
    return list(
        set(
            [edge[1] for edge in edges if edge[0]==node] + [edge[0] for edge in edges if edge[1]==node]
            ))


def get_highest_contrast_neighbor(g, node, neighbors):
    """ TODO """
    c1 = g.nodes[node]['mean color']
    highest_contrast_neighbor = None
    highest_contrast = -np.inf
    for neighbor in neighbors:
        c2 = g.nodes[neighbor]['mean color']
        dist = distance.euclidean(c1, c2)
        if dist > highest_contrast:
            highest_contrast_neighbor = neighbor
            highest_contrast = dist                    
    return highest_contrast_neighbor




class SuperpixelDataset(torch.utils.data.Dataset):
    """Invariant dataset."""
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
        # 'convert' to rgb
        img_rgb = np.zeros((28,28,3))
        img_rgb[:,:,2] = img
        img_rgb[:,:,1] = img
        img_rgb[:,:,0] = img


        img = np.float32(np.asarray(img[0,:,:]))/255
        labels = slic(img_rgb, n_segments = 15, sigma = 0.1, start_label=0)
        p = regionprops(labels+1,intensity_image=img)
        g = graph.rag_mean_color(img, labels)

        feats = []
        coords = []
     
        edge_locations = []
        edge_orientations = []

        for node in g.nodes:
            neighbors = get_neighbors(node, g.edges)
            highest_contrast_neighbor = get_highest_contrast_neighbor(g, node, neighbors)
            edge_center = (np.array(p[node].centroid) + np.array(p[highest_contrast_neighbor].centroid))/2

            edge_orientation = np.rad2deg(
                np.arctan2(
                    p[highest_contrast_neighbor].centroid[-1] - p[highest_contrast_neighbor].centroid[0],
                    p[node].centroid[-1] - p[node].centroid[0]
                    ))
            
            edge_locations.append(torch.Tensor(np.array(edge_center)))
            # edge_locations.append(torch.Tensor(np.array(p[node].centroid)))
            edge_orientations.append(edge_orientation)

            color = p[node]['mean_intensity']
            orientation = p[node].orientation#['orientation']
            invariants = p[node]['moments_hu']
            center = torch.Tensor(p[node]['centroid']).unsqueeze(0)
            feat = torch.cat([torch.Tensor([color]),torch.Tensor(invariants)]).unsqueeze(0)
            feats.append(feat)
            coords.append(center)

        edge_locations = torch.stack(edge_locations, dim=0)#.unsqueeze(0)
        edge_orientations = torch.Tensor(np.array(edge_orientations))#.unsqueeze(0)
        feats = torch.cat(feats,dim=0)
        coords = torch.cat(coords,dim=0)


        # print(coords)
        return dict(
            img_rgb=img_rgb,
            locations=edge_locations,
            orientations=edge_orientations,
            target=target,
            segments=labels,
            )




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


trainloader = torch.utils.data.DataLoader(mnist_train, batch_size=10, shuffle=True, collate_fn=collate_edges)
sample = next(iter(trainloader))


print(sample.keys())

fig, ax = plt.subplots()
ax.imshow(mark_boundaries(sample['img'][0], sample['segments'][0]))
for i in range(sample['locations'][0].size(0)):
    loc = sample['locations'][0][i]
    orientation = sample['orientations'][0][i]
    circ = Circle(loc, 0.3, color='red')
    slope = np.tan(orientation.numpy())
    plot_line(ax, loc, slope)
    ax.add_patch(circ)
plt.show()