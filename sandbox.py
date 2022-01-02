import numpy as np
import torch
from torchvision import datasets, transforms
from types import SimpleNamespace
import pandas as pd
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

from models import EGNN, get_edges_batch



def collate_edges(batch):
    """ TODO docstring"""
    batch_imgs = [b['img_rgb'] for b in batch]
    batch_edges = [b['edges'] for b in batch]
    batch_edges_rows = [b['edges'][0] for b in batch]
    batch_edges_columns = [b['edges'][1] for b in batch]
    batch_locations = [b['locations'] for b in batch]
    batch_orientations = [b['orientations'] for b in batch]
    batch_targets = [b['target'] for b in batch]
    batch_segments = [b['segments'] for b in batch]
    # ###############################################
    node_centers=[b['node_centers'] for b in batch]

    batch_edges_rows = torch.nn.utils.rnn.pad_sequence(batch_edges_rows, batch_first=True)
    batch_edges_columns = torch.nn.utils.rnn.pad_sequence(batch_edges_columns, batch_first=True)
    batch_locations = torch.nn.utils.rnn.pad_sequence(batch_locations,batch_first=True)
    batch_orientations = torch.nn.utils.rnn.pad_sequence(batch_orientations,batch_first=True)
    node_centers = torch.nn.utils.rnn.pad_sequence(node_centers, batch_first=True)
    return dict(
                img=batch_imgs,
                edges=[batch_edges_rows, batch_edges_columns],
                locations=batch_locations,
                orientations=batch_orientations,
                target=batch_targets,
                segments=batch_segments,
                ########################
                node_centers=node_centers
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
        labels = slic(img_rgb, n_segments = 10, sigma = 0.1, start_label=0)
        # labels = slic(img_rgb, n_segments = 30, sigma = 0.1, start_label=0)
        p = regionprops(labels+1,intensity_image=img)
        g = graph.rag_mean_color(img_rgb, labels)
        # g = graph.rag_mean_color(img, labels)

        feats = []
        coords = []
     
        edge_locations = []
        edge_orientations = []

        rows, columns = [], []

        node_centers = []

        meta_data = {}


        for node in g.nodes:
            neighbors = get_neighbors(node, g.edges)


            highest_contrast_neighbor = get_highest_contrast_neighbor(g, node, neighbors)


            rows.append(node)
            columns.append(highest_contrast_neighbor)

            edge_center = (np.array(p[node].centroid) + np.array(p[highest_contrast_neighbor].centroid))/2
            edge_center = edge_center[::-1] # conform to x, y ordering

            edge_orientation = np.rad2deg(
                np.arctan2(
                    p[highest_contrast_neighbor].centroid[-1] - p[highest_contrast_neighbor].centroid[0],
                    p[node].centroid[-1] - p[node].centroid[0]
                    ))
            
            edge_locations.append(torch.Tensor(np.array(edge_center)))
            edge_orientations.append(edge_orientation)

            color = p[node]['mean_intensity']
            orientation = p[node].orientation
            invariants = p[node]['moments_hu']


            center = torch.Tensor(p[node]['centroid'][::-1]).unsqueeze(0)
            feat = torch.cat([torch.Tensor([color]),torch.Tensor(invariants)]).unsqueeze(0)
            feats.append(feat)
            coords.append(center)


            node_centers.append(center)



            meta_data[str(node)] = dict(
                center=center,
                color=color,
                invariants=invariants,
                highest_contrast_neighbor=highest_contrast_neighbor,
                )
            # node_centers.append(p[node].centroid)


        edges = torch.stack((torch.LongTensor(rows), torch.LongTensor(columns)))
        edge_locations = torch.stack(edge_locations, dim=0)
        edge_orientations = torch.Tensor(np.array(edge_orientations))

        feats = torch.cat(feats,dim=0)
        coords = torch.cat(coords,dim=0)
        node_centers = torch.cat(node_centers,dim=0)

        return dict(
            img_rgb=img_rgb,
            edges=edges,
            locations=edge_locations,
            orientations=edge_orientations,
            target=target,
            segments=labels,
            meta_data=pd.DataFrame.from_dict(meta_data, orient='index'),
            ##########
            node_centers=node_centers
            )






def train_single_epoch(model, dataloader, testloader, optimizer, loss_function):
    model.train()
    pbar = tqdm(total=len(trainloader),position=0, leave=True)

    for batch_idx, sample in enumerate(trainloader):
        x = sample['locations'].view(-1,2)
        h = sample['orientations'].view(-1,1)
        edges, edge_attr = get_edges_batch(sample['edges'][0][1].size(0), batch_size)
        
        edges = [sample['edges'][0].flatten(),sample['edges'][1].flatten()]
        edge_attr = edge_attr[:edges[0].size(0)]

        out_h, out_x = model(h, x, edges, edge_attr)
        scores = out_h.view(batch_size, sample['edges'][0][1].size(0), -1).mean(1)#.mean(1)
        target = torch.Tensor(sample['target']).long()
        loss = loss_function(scores, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        pbar.set_description(f"Training loss: {loss.item():.2f}, Accuracy: {(scores.max(1)[1]==target).float().mean().item():.2f}")# % (loss.item(),( ) )
        pbar.update()
    pbar.close()
    accuracy = evaluate(model, testloader)

    optimizer.param_groups[0]['lr'] *= 0.5

    return accuracy


def evaluate(model, testloader):
    model.eval()
    with torch.no_grad():
        total_correct = 0
        for batch_idx, sample in enumerate(testloader):
            x = sample['locations'].view(-1,2)
            h = sample['orientations'].view(-1,1)
            edges, edge_attr = get_edges_batch(sample['edges'][0][1].size(0), 10)
            
            edges = [sample['edges'][0].flatten(),sample['edges'][1].flatten()]
            edge_attr = edge_attr[:edges[0].size(0)]

            out_h, out_x = model(h, x, edges, edge_attr)
            scores = out_h.view(10, sample['edges'][0][1].size(0), -1).mean(1)#.mean(1)
            target = torch.Tensor(sample['target']).long()
            # loss = loss_function(scores, target)

            _, predicted = torch.max(scores.data, 1)
            total_correct += (predicted == target).sum().item() / 10

        accuracy = (100 * total_correct) / len(testloader)
        return accuracy


if __name__ == '__main__':
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


    trainloader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True, collate_fn=collate_edges)
    testloader = torch.utils.data.DataLoader(mnist_test, batch_size=10, shuffle=True, collate_fn=collate_edges)



    sample = next(iter(trainloader))
    print(sample.keys())



    # model = EGNN(in_node_nf=1, hidden_nf=32, out_node_nf=10, in_edge_nf=1)
    # # # Dummy parameters
    # batch_size = 100
    # n_nodes = sample['edges'][0][1].size(0)
    # n_feat = 1
    # x_dim = 2


    # # fake edges, edge_attributes
    # edges, edge_attr = get_edges_batch(n_nodes, batch_size)
    # edges = [sample['edges'][0].flatten(),sample['edges'][1].flatten()]

    # # reduce size of fake edge attributes so it matches real edges
    # edge_attr = edge_attr[:edges[0].size(0)]


    # loss_function = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)
    # # optimizer = torch.optim.SGD(model.parameters(),lr=1e-4)




    # # print(model)
    # for epoch in range(3):
    #     epoch_accuracy = train_single_epoch(model, trainloader, testloader, optimizer, loss_function)
    #     print(f"Accuracy after epoch {epoch}: {epoch_accuracy:.2f}")




    # # # Run EGNN
    # # h, x = egnn(h, x, edges, edge_attr)
    # # out_feats,out_coords=net(feats, coords, edges, edge_attr=None)


    fig, ax = plt.subplots()
    ax.imshow(mark_boundaries(sample['img'][0], sample['segments'][0]))
    for i in range(len(sample['node_centers'][0])):
        loc = sample['node_centers'][0][i].cpu().numpy()
        circ = Circle(loc, 0.3, color='red')
        # y, x = loc
        # print(x)
        # print(y)
        # print()
        # circ = Circle((x,y), 0.3, color='red')
        ax.add_patch(circ)

    for i in range(sample['locations'][0].size(0)):
        loc = sample['locations'][0][i]
        # orientation = sample['orientations'][0][i]
        circ = Circle(loc, 0.3, color='green')
        # slope = np.tan(orientation.numpy())
        # plot_line(ax, loc, slope)
        ax.add_patch(circ)
    plt.show()