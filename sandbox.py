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
import math
import pickle

from models import EGNN, get_edges_batch
from utils import get_bearing, get_highest_contrast_neighbor, get_neighbors, plot_line


def collate(batch):
    """ TODO docstring"""
    meta_data = [d['meta_data'].loc[d['meta_data']['is_darker']==1] for d in batch]
    batch_locations = [np.array(list(meta_datum['center_between'])) for meta_datum in meta_data]
    batch_angles = [meta_datum['angle_between'].to_numpy() for meta_datum in meta_data]
    # batch_nodes_dark = [meta_datum.index.astype(np.int32).to_numpy() for meta_datum in meta_data]
    # batch_nodes_bright = [meta_datum['highest_contrast_neighbor'].to_numpy() for meta_datum in meta_data]
    batch_colors = [meta_datum['highest_contrast_neighbor_color'].to_numpy() for meta_datum in meta_data]
    batch_invariants = [[moment for moment in meta_datum['highest_contrast_neighbor_invariants'].to_numpy()] for meta_datum in meta_data]


    # convert and pad locations
    batch_locations = torch.nn.utils.rnn.pad_sequence(
        [torch.from_numpy(locations) for locations in batch_locations],
        batch_first=True
        )

    batch_size = batch_locations.size(0)

    # concatenate orientations, colors and invariants of darker nodes to form features
    batch_feats = []
    for i in range(len(batch)):
        angles = torch.from_numpy(batch_angles[i]).unsqueeze(1)
        colors = torch.from_numpy(batch_colors[i]).unsqueeze(1)
        invariants = torch.from_numpy(np.array(batch_invariants[i]))
        feats = torch.cat((angles, colors, invariants), dim=-1)
        batch_feats.append(feats)

    # pad features
    batch_feats = torch.nn.utils.rnn.pad_sequence(batch_feats, batch_first=True).to(torch.float32)
    # pad 
    # node_centers=[b['node_centers'] for b in batch]
    # node_centers = torch.nn.utils.rnn.pad_sequence(node_centers, batch_first=True).to(torch.float32)

    n_nodes = batch_feats.size(1)
    batch_size = batch_feats.size(0)
    # print('regular')
    # print(n_nodes)
    # print(batch_size)
    edges, _ = get_edges_batch(n_nodes, batch_size)

    return dict(
                # for model
                edges=edges,
                locations = (batch_locations.view(-1,2) / 28).to(torch.float32),
                feats = batch_feats.view(-1,9).to(torch.float32),
                target=torch.LongTensor([b['target'] for b in batch]),
                n_nodes = n_nodes,
                batch_size=batch_size,
                # for plotting
                meta_data=meta_data,
                segments=[b['segments'] for b in batch],
                img=[b['img_rgb'] for b in batch],
                # node_centers=node_centers
        )

def collate_graph(batch):
    """ TODO docstring"""
    # meta_data = [d['meta_data'].loc[d['meta_data']['is_darker']==1] for d in batch]
    batch_locations = [np.array(datum['locations']) for datum in batch]
    batch_orientations = [np.array(datum['orientations']) for datum in batch]
    batch_sins = [np.array(datum['sins']) for datum in batch]
    batch_cossins = [np.array(datum['cossins']) for datum in batch]
    batch_colors = [np.array(datum['colors']) for datum in batch]
    batch_invariants = [np.array(datum['invariants']) for datum in batch]
    batch_targets = [datum['target'] for datum in batch]
    batch_img_paths = [datum['img_path'] for datum in batch]
    # print(batch_targets)
    # print(batch_locations[0].shape)

    # batch_angles = [meta_datum['angle_between'].to_numpy() for meta_datum in meta_data]
    # batch_nodes_dark = [meta_datum.index.astype(np.int32).to_numpy() for meta_datum in meta_data]
    # batch_nodes_bright = [meta_datum['highest_contrast_neighbor'].to_numpy() for meta_datum in meta_data]
    # batch_colors = [meta_datum['highest_contrast_neighbor_color'].to_numpy() for meta_datum in meta_data]
    # batch_invariants = [[moment for moment in meta_datum['highest_contrast_neighbor_invariants'].to_numpy()] for meta_datum in meta_data]


    # # convert and pad locations
    batch_locations = torch.nn.utils.rnn.pad_sequence(
        [torch.from_numpy(locations) for locations in batch_locations],
        batch_first=True
        )

    # batch_size = batch_locations.size(0)

    # # concatenate orientations, colors and invariants of darker nodes to form features
    batch_feats = []
    for i in range(len(batch)):
        angles = torch.from_numpy(batch_orientations[i]).unsqueeze(1)
        sin = torch.from_numpy(batch_sins[i]).unsqueeze(1)
        cos = torch.from_numpy(batch_cossins[i]).unsqueeze(1)
        colors = torch.from_numpy(batch_colors[i]).unsqueeze(1)
        invariants = torch.from_numpy(np.array(batch_invariants[i]))
        feats = torch.cat((angles, sin, cos, invariants), dim=-1)
        batch_feats.append(feats)

    # # pad features
    batch_feats = torch.nn.utils.rnn.pad_sequence(batch_feats, batch_first=True).to(torch.float32)
    # pad 
    # # node_centers=[b['node_centers'] for b in batch]
    # # node_centers = torch.nn.utils.rnn.pad_sequence(node_centers, batch_first=True).to(torch.float32)


    # print(batch_feats.size())

    batch_size = batch_feats.size(0)
    n_nodes = batch_feats.size(1)
    feature_size = batch_feats.size(2)
    # print('graph')
    # print(n_nodes)
    # print(batch_size)

    edges, _ = get_edges_batch(n_nodes, batch_size)

    return dict(
                # locations=batch_locations,
                locations = (batch_locations.view(-1,2) / 28).to(torch.float32),
                feats = batch_feats.view(-1, feature_size).to(torch.float32),
                edges = edges,
                batch_size=batch_size,
                n_nodes=n_nodes,
                feature_size = feature_size,
                img_paths=batch_img_paths,
                # edges=edges,
                target=torch.LongTensor([b['target'] for b in batch]),
                # n_nodes = n_nodes,
                # batch_size=batch_size,
                # for plotting
                # meta_data=meta_data,
                # segments=[b['segments'] for b in batch],
                # img=[b['img_rgb'] for b in batch],
                # node_centers=node_centers
        )


class GraphMNIST(torch.utils.data.Dataset):
    def __init__(self, data_root='./graph_mnist/',is_train=True):
        self.data_root = data_root
        self.is_train = is_train
        # dtype_lookup = dtype={'center':np.ndarray, 'color':np.ndarray, 'invariants':np.ndarray}
        if self.is_train:
            with open(data_root + 'train_data.pickle', "rb") as fp:
                self.data_dict = pickle.load(fp)
        elif not self.is_train:
            with open(data_root + 'test_data.pickle', "rb") as fp:
                self.data_dict = pickle.load(fp)

        self.data = pd.DataFrame(self.data_dict)
        # print(self.data.head())

    def __len__(self):
        return self.data.shape[0]
        # return 100
    def __getitem__(self, ix):
        sample = self.data.iloc[ix]

        mask = sample['is_darker']

        locations = []
        orientations = []
        sins, cossins = [], []
        colors = []
        invariants = []

        for i in range(len(mask)):
            if mask[i] == 1:
                locations.append(sample['center_between'][i])

                orientations.append(sample['angle_between'][i])
                sins.append(sample['sin'][i])
                cossins.append(sample['cos'][i])

                colors.append(sample['color'][i])
                invariants.append(sample['invariants'][i])

        target = sample['targets']


        return dict(
            locations=locations,
            orientations=orientations,
            sins=sins,
            cossins = cossins,
            invariants=invariants,
            colors=colors,
            target=target,
            img_path=sample['img_paths']
            )

        # print(sample.keys())
        # batch_nodes_dark = [meta_datum.index.astype(np.int32).to_numpy() for meta_datum in meta_data]
        # batch_nodes_bright = [meta_datum['highest_contrast_neighbor'].to_numpy() for meta_datum in meta_data]
        # batch_colors = [meta_datum['highest_contrast_neighbor_color'].to_numpy() for meta_datum in meta_data]
        # batch_invariants = [[moment for moment in meta_datum['highest_contrast_neighbor_invariants'].to_numpy()] for meta_datum in meta_data]



        # return dict(
        #     img_rgb=img_rgb,
        #     locations=edge_locations,
        #     orientations=edge_orientations,
        #     target=target,
        #     segments=labels,
        #     meta_data=pd.DataFrame.from_dict(meta_data, orient='index'),
        #     ##########
        #     node_centers=node_centers
        #     )



        # return torch.ones(3,3)


class SuperpixelEdgesDataset(torch.utils.data.Dataset):
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
        labels = slic(img_rgb, n_segments = 25, compactness = 50, sigma = 0.1, start_label=0)
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
                center=p[node].centroid[::-1],
                color=color,
                invariants=invariants,
                is_darker=(color <= p[highest_contrast_neighbor].mean_intensity).astype(np.int32),
                highest_contrast_neighbor=highest_contrast_neighbor,
                highest_contrast_neighbor_loc=p[highest_contrast_neighbor].centroid[::-1],
                highest_contrast_neighbor_color=p[highest_contrast_neighbor].mean_intensity,
                highest_contrast_neighbor_invariants=p[highest_contrast_neighbor].moments_hu,
                angle_between=get_bearing(p[node].centroid[::-1], p[highest_contrast_neighbor].centroid[::-1]),
                center_between=edge_center,
                )


        edge_locations = torch.stack(edge_locations, dim=0)
        edge_orientations = torch.Tensor(np.array(edge_orientations))

        feats = torch.cat(feats,dim=0)
        coords = torch.cat(coords,dim=0)
        node_centers = torch.cat(node_centers,dim=0)

        return dict(
            img_rgb=img_rgb,
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

    graph_mnist_train = GraphMNIST()
    graph_mnist_test = GraphMNIST(is_train=False)
    # print(len(graph_mnist_test))

    # trainloader = torch.utils.data.DataLoader(graph_mnist_train, batch_size=16, shuffle=True, collate_fn=collate)
    graph_trainloader = torch.utils.data.DataLoader(graph_mnist_train, batch_size=16, shuffle=True, collate_fn=collate_graph)
    graph_testloader = torch.utils.data.DataLoader(graph_mnist_test, batch_size=1, shuffle=False, collate_fn=collate_graph)


    graph_sample = next(iter(graph_trainloader))
    print(graph_sample.keys())
    # test_sample = next(iter(graph_testloader))
    # print(graph_sample['locations'].size())
    # print(test_sample['locations'].size())

    # print(graph_sample['locations'].size())
    # print(graph_sample['feats'].size())
    # print(graph_sample['edges'][0].size())



    # new features:
    # colors of both darker and brigher
    # sine and cosine of orientation




    # total_transform = transforms.Compose([
    #                 #RandomReduceScale(0.3,1),
    #                 #RandomRotate(-180,180),
    #                 transforms.ToTensor()])
    # # total_transform_test = transforms.Compose([
    # #                 #RandomReduceScale(0.3,1),
    # #                 #RandomRotate(-180,180),
    # #                 transforms.ToTensor()])




    # mnist_train = SuperpixelEdgesDataset(datasets.MNIST('data', train=True, download=True,transform=total_transform))
    # mnist_test = SuperpixelEdgesDataset(datasets.MNIST('data', train=False, download=True,transform=total_transform_test))


    # trainloader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True, collate_fn=collate_edges)
    # # testloader = torch.utils.data.DataLoader(mnist_test, batch_size=10, shuffle=True, collate_fn=collate_edges)

    # trainloader = torch.utils.data.DataLoader(mnist_train, batch_size=16, shuffle=True, collate_fn=collate)
    # # testloader = torch.utils.data.DataLoader(mnist_test, batch_size=10, shuffle=True, collate_fn=collate_edges)




    # sample = next(iter(trainloader))

    # print(sample.keys())
    # print('-------')
    # print(sample['locations'].size())
    # print(sample['feats'].size())
    # print(sample['edges'][0].size())
    
    # print('------')
    # feats = sample['feats']
    # edges = sample['edges']
    # n_nodes = sample['n_nodes']
    # batch_size = sample['batch_size']




    # feats = graph_sample['feats']
    # edges = graph_sample['edges']
    # coords = graph_sample['locations']
    # n_nodes = graph_sample['n_nodes']
    # batch_size = graph_sample['batch_size']

    # n_feat = feats.size(1)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    # model = EGNN(
    #     in_node_nf=9, # 9
    #     hidden_nf=100,
    #     out_node_nf=10,
    #     in_edge_nf=0,
    #     attention=True,
    #     normalize=True,
    #     n_layers=6,
    # ).to(device)

    # out_feats, out_coords = model(feats, coords, edges, edge_attr=None)
    # scores = out_feats.view(batch_size,n_nodes,-1).mean(1)
    # print(scores.size())
    # print(scores)


    # loss_function = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)


    # for epoch in range(5):
    #     pbar = tqdm(total=len(graph_trainloader),position=0, leave=True)
    #     for _, subsample in enumerate(graph_trainloader):
    #         feats = graph_sample['feats']
    #         edges = graph_sample['edges']
    #         coords = graph_sample['locations']
    #         n_nodes = graph_sample['n_nodes']
    #         batch_size = graph_sample['batch_size']
    #         target = graph_sample['target']

    #         # n_feat = feats.size(1)

    #         for e in range(len(edges)):
    #             edges[e] = edges[e].to(device)
    #         feats, coords, target = feats.to(device), coords.to(device), target.to(device)
    #         out_feats,out_coords=model(feats, coords, edges, edge_attr=None)
    #         scores = out_feats.view(batch_size,n_nodes,-1).mean(1)

    #         loss = loss_function(scores,target)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         pbar.set_description("Training loss: %f, Class: %f" % (loss.item(),(scores.max(1)[1]==target).float().mean().item() ) )
    #         pbar.update()
    #     pbar.close()
    #     optimizer.param_groups[0]['lr'] *= 0.5





# (batch_feats, batch_coords, target, edges,n_nodes,batch_size)
    # feats, coords, target = batch_feats.to(device), batch_coords.to(device), target.to(device)
    # print(feats.size())
    # out_feats,out_coords=net(feats, coords, edges, edge_attr=None)
    # print(out_feats.size())
    # print()
    # print(sample['locations'].size())
    # print(sample['feats'].size())
    # print(sample['target'].size())
    # print(sample['edges'][0].size())


    # print(sample['edges'][0][0])
    # print(sample['meta_data'][0][['center', 'highest_contrast_neighbor', 'highest_contrast_neighbor_loc']])
    # print(sample['meta_data'][0][['is_darker']])



    # fig, ax = plt.subplots()
    # ax.imshow(mark_boundaries(sample['img'][0], sample['segments'][0],mode='inner'))
    # for index, data in sample['meta_data'][0].iterrows():
    # #     # print(row.center)
    # #     # loc = sample['node_centers'][0][i].cpu().numpy()
    #     node_circ = Circle(data.center, 0.3, color='red')
    #     ax.plot(
    #         [data.center[0], data.highest_contrast_neighbor_loc[0]],
    #         [data.center[1], data.highest_contrast_neighbor_loc[1]],
    #         color='blue'
    #         )
    #     ax.add_patch(node_circ)

    #     neighbor_circ = Circle(data.center_between, 0.3, color='green')
    #     ax.add_patch(neighbor_circ)

    # plt.show()

    # # print(angle_between((1, 0), (0, 1)))



    

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


    # fig, ax = plt.subplots()
    # ax.imshow(mark_boundaries(sample['img'][0], sample['segments'][0],mode='inner'))
    # for i in range(len(sample['node_centers'][0])):
    #     loc = sample['node_centers'][0][i].cpu().numpy()
    #     circ = Circle(loc, 0.3, color='red')
    #     # y, x = loc
    #     # print(x)
    #     # print(y)
    #     # print()
    #     # circ = Circle((x,y), 0.3, color='red')
    #     ax.add_patch(circ)

    # for i in range(sample['locations'][0].size(0)):
    #     loc = sample['locations'][0][i]
    #     # orientation = sample['orientations'][0][i]
    #     circ = Circle(loc, 0.3, color='green')
    #     # slope = np.tan(orientation.numpy())
    #     # plot_line(ax, loc, slope)
    #     ax.add_patch(circ)
    # plt.show()