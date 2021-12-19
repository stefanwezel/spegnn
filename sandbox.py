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




def get_neighbors(node, edges):
    # print(node)
    # print(edges)

    return list(set([edge[1] for edge in edges if edge[0]==node] + [edge[0] for edge in edges if edge[1]==node]))


def get_highest_contrast_neighbor(g, node, neighbors):
    
    c1 = g.nodes[node]['mean color']
    highest_contrast_neighbor = None
    highest_contrast = -np.inf
    for neighbor in neighbors:

        c2 = g.nodes[neighbor]['mean color']
        dist = distance.euclidean(c1, c2)
        if dist >= highest_contrast:
            highest_contrast_neighbor = neighbor
            highest_contrast = dist                    
    return highest_contrast_neighbor




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
        # x.unsqueeze_(0)
        # img = data[s,0,:,:].numpy()
        # print(img.shape)

        # print(img.min())

        img_rgb = np.zeros((28,28,3))
        img_rgb[:,:,2] = img
        img_rgb[:,:,1] = img
        img_rgb[:,:,0] = img

        # print(img_rgb[:,:,0] == img_rgb[:,:,1])
        # print(img_rgb.shape)


        img = np.float32(np.asarray(img[0,:,:]))/255
        # img_rgb = Image.fromarray(img.astype(int), 'RGB')
        # print(img_rgb.size)
        # print(img.shape)
        # img_rgb = torch.from_numpy(img).unsqueeze(0)
        # img_rgb = img_rgb.repeat(3, 1, 1)
        # img_rgb = img_rgb.reshape(28,28,3)
        # print(img_rgb.size())

        labels = slic(img_rgb, n_segments = 15, sigma = 0.1, start_label=0)



        p = regionprops(labels+1,intensity_image=img)
        g = graph.rag_mean_color(img, labels)
        # print(g)
        # print(g.nodes[0].keys())
        # print(g.nodes[0]['labels'][0])
        # print(g.edges)
        # print(get_neighbors(g.nodes[0]['labels'][0], g.edges))
        # print(g.neighbors(5))
        # print(dir(g))

        feats = []
        coords = []
     

        for node in g.nodes:
            neighbors = get_neighbors(node, g.edges)
            highest_contrast_neighbor = get_highest_contrast_neighbor(g, node, neighbors)
            print(highest_contrast_neighbor)
            print()

            
            color = p[node]['mean_intensity']
            # print(dir(g.neighbors(node)))
            # print(color)
            orientation = p[node].orientation#['orientation']
            # print(orientation)
            invariants = p[node]['moments_hu']

            # print(p[node]['weighted_moments_hu'])
            # print(dir(p[node]))
            center = torch.Tensor(p[node]['centroid']).unsqueeze(0)
            feat = torch.cat([torch.Tensor([color]),torch.Tensor(invariants)]).unsqueeze(0)
            feats.append(feat)
            coords.append(center)


            # print(dir(p[node]))

        # print(g)
        # print(p)
        # print()

        locations = []
        orientations = []


        for edge in list(g.edges)[:self.num_edges]: # 'cropping'
            node1 = edge[0]
            node2 = edge[1]


            # print(p[node1]['mean_intensity'])

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

        return dict(img=img, locations=locations, orientations=orientations, target=target, labels=labels, img_rgb=img_rgb)




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





trainloader = torch.utils.data.DataLoader(mnist_train, batch_size=1, shuffle=True)


sample = next(iter(trainloader))







# image = sample['img_rgb'][0]
# backtorgb = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
# img_rgb = np.zeros((img.shape[0], img.shape[1], 3))
# for d in range(3):
#     img_rgb[:,:,d] = img
# print(img_rgb.shape)
# fig, ax = plt.subplots(2)
# image = img_as_float(io.imread('/home/stefan/Downloads/mnist_2_small.png'))[:,:,:3]
# image = cv2.imread('image.png')
# segments = slic(image, n_segments = 15, sigma = 0.1, start_label=0)
# plt.imshow(sample['img'][0])
# print(segments==sample['labels'][0].numpy())
# print()



# plt.imshow(mark_boundaries(sample['img_rgb'][0], sample['labels'][0].numpy()))
# plt.show()

# plt.imshow(mark_boundaries(sample['img'][0], sample['labels'][0].numpy()), alpha=0.3)
# ax[0].imshow(sample['img'][0], cmap='Blues_r')

# for i in range(sample['locations'][0].size(0)):
#     loc = sample['locations'][0][i]
#     orientation = sample['orientations'][0][i]
#     circ = Circle(loc, 0.3, color='red')
#     slope = np.tan(orientation.numpy())

#     plot_line(ax[0], loc, slope)
#     ax[0].add_patch(circ)


# ax[0].imshow(mark_boundaries(sample['img'][0], sample['labels'][0].numpy()))


# # segments = slic(sample['img'][0], n_segments=25, compactness=0.5, sigma=0.1, start_label=1)
# # ax[0].imshow(mark_boundaries(sample['img'][0], segments))
# plt.show()
