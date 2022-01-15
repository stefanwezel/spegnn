import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
from skimage.segmentation import slic
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



class SuperpixelImgDataset(torch.utils.data.Dataset):
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

        # print(idx)

        img,target = self.orig_dataset[idx]
        # 'convert' to rgb
        img_rgb = np.zeros((28,28,3))
        img_rgb[:,:,2] = img
        img_rgb[:,:,1] = img
        img_rgb[:,:,0] = img

        img = np.float32(np.asarray(img[0,:,:]))/255
        labels = slic(img_rgb, n_segments = 25, compactness = 50, sigma = 0.1, start_label=0)
        # labels = slic(img_rgb, n_segments = 30, sigma = 0.1, start_label=0)
        p = regionprops(labels+1,intensity_image=img)
        g = graph.rag_mean_color(img_rgb, labels)

        meta_data = {}
        # meta_data['target'] = target


        for node in g.nodes:
            neighbors = get_neighbors(node, g.edges)
            highest_contrast_neighbor = get_highest_contrast_neighbor(g, node, neighbors)

            # edge_center = (np.array(p[node].centroid) + np.array(p[highest_contrast_neighbor].centroid))/2
            # edge_center = edge_center[::-1] # conform to x, y ordering

            

            # color = p[node]['mean_intensity']
            # orientation = p[node].orientation
            # invariants = p[node]['moments_hu']


            # center = torch.Tensor(p[node]['centroid'][::-1]).unsqueeze(0)

            meta_data[str(node)] = dict(
                center=p[node].centroid[::-1],
                color=p[node]['mean_intensity'],
                invariants=p[node]['moments_hu'],
                is_darker=(p[node]['mean_intensity'] <= p[highest_contrast_neighbor].mean_intensity).astype(np.int32),
                highest_contrast_neighbor=highest_contrast_neighbor,
                highest_contrast_neighbor_loc=p[highest_contrast_neighbor].centroid[::-1],
                highest_contrast_neighbor_color=p[highest_contrast_neighbor].mean_intensity,
                highest_contrast_neighbor_invariants=p[highest_contrast_neighbor].moments_hu,
                angle_between=get_bearing(p[node].centroid[::-1], p[highest_contrast_neighbor].centroid[::-1]),
                center_between=((np.array(p[node].centroid) + np.array(p[highest_contrast_neighbor].centroid))/2)[::-1],
                )



        return dict(
                img=img_rgb,
                centers=[meta_data[node]['center'] for node in meta_data.keys()],
                colors=[meta_data[node]['color'] for node in meta_data.keys()],
                invariants=[meta_data[node]['invariants'] for node in meta_data.keys()],
                is_darkers=[meta_data[node]['is_darker'] for node in meta_data.keys()],
                highest_contrast_neighbors=[meta_data[node]['highest_contrast_neighbor'] for node in meta_data.keys()],
                highest_contrast_neighbor_locs=[meta_data[node]['highest_contrast_neighbor_loc'] for node in meta_data.keys()],
                highest_contrast_neighbor_colors=[meta_data[node]['highest_contrast_neighbor_color'] for node in meta_data.keys()],
                highest_contrast_neighbor_invariants=[meta_data[node]['highest_contrast_neighbor_invariants'] for node in meta_data.keys()],
                angle_betweens=[meta_data[node]['angle_between'] for node in meta_data.keys()],
                center_betweens=[meta_data[node]['center_between'] for node in meta_data.keys()],
                target=target,
                # meta_data=meta_data,
            )




import time 
total_transform = transforms.Compose([
                transforms.ToTensor()])

mnist_train = SuperpixelImgDataset(datasets.MNIST('data', train=True, download=True,transform=total_transform))
mnist_test = SuperpixelImgDataset(datasets.MNIST('data', train=False, download=True,transform=total_transform))




# trainloader = torch.utils.data.DataLoader(mnist_train, batch_size=16, shuffle=True)
# testloader = torch.utils.data.DataLoader(mnist_test, batch_size=10, shuffle=True)



centers = []
colors = []
invariants = []
is_darkers = []
highest_contrast_neighbors = []
highest_contrast_neighbor_locs = []
highest_contrast_neighbor_colors = []
highest_contrast_neighbor_invariants = []
angle_betweens = []
center_betweens = []
targets = []
img_paths = []
is_trains = []
file_names = []


# start_time = time.time()
# for i, sample in enumerate(mnist_train):
for i, sample in enumerate(mnist_test):
    # print(i)
    # if i < 6000:
    if i < 1000:
        # print(sample['img'].shape)
        # img_name = f"./graph_mnist/train_imgs/mnist_{i:05d}.npy"
        img_name = f"./graph_mnist/test_imgs/mnist_{6000+i:05d}.npy"
        np.save(img_name, sample['img'])
        file_names.append(img_name)
        lens = [len(sample[key]) for key in ['centers', 'colors', 'invariants', 'is_darkers', 'highest_contrast_neighbors', 'highest_contrast_neighbor_locs', 'highest_contrast_neighbor_colors', 'highest_contrast_neighbor_invariants', 'angle_betweens', 'center_betweens']]
        assert len(set(lens)) == 1 # check whether all elements have same lengths
        # print(lens)
        # len(set(input_list))! =1:
        centers.append(sample['centers'])
        colors.append(sample['colors'])
        invariants.append(sample['invariants'])
        is_darkers.append(sample['is_darkers'])
        highest_contrast_neighbors.append(sample['highest_contrast_neighbors'])
        highest_contrast_neighbor_locs.append(sample['highest_contrast_neighbor_locs'])
        highest_contrast_neighbor_colors.append(sample['highest_contrast_neighbor_colors'])
        highest_contrast_neighbor_invariants.append(sample['highest_contrast_neighbor_invariants'])
        angle_betweens.append(sample['angle_betweens'])
        center_betweens.append(sample['center_betweens'])
        targets.append(sample['target'])
        img_paths.append(img_name)
        is_trains.append(0)


    else:
        break



data = dict(
    center=centers,
    color=colors,
    invariants=invariants,
    is_darker=is_darkers,
    highest_contrast_neighbor=highest_contrast_neighbors,
    highest_contrast_neighbor_loc=highest_contrast_neighbor_locs,
    highest_contrast_neighbor_color=highest_contrast_neighbor_colors,
    highest_contrast_neighbor_invariants=highest_contrast_neighbor_invariants,
    angle_between=angle_betweens,
    sin=[np.sin(np.radians(angle)) for angle in angle_betweens],
    cos=[np.cos(np.radians(angle)) for angle in angle_betweens],
    center_between=center_betweens,
    targets=targets,
    img_paths=img_paths,
    is_train=is_trains,
    file_name=file_names,
    )




# with open("./graph_mnist/train_data.pickle", "wb") as fp:
#     pickle.dump(data, fp)
with open("./graph_mnist/test_data.pickle", "wb") as fp:
    pickle.dump(data, fp)




# df = pd.DataFrame(data)
# print(df)
# print(df.shape)
# print(df.columns)
# df.to_csv('./graph_mnist/test_data.csv')



# total_time = time.time() - start_time
# print(total_time)