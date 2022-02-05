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

def collate_graph(batch):
    """ TODO docstring"""
    # meta_data = [d['meta_data'].loc[d['meta_data']['is_darker']==1] for d in batch]
    batch_locations = [np.array(datum['locations']) for datum in batch]
    batch_orientations = [np.array(datum['orientations']) for datum in batch]
    batch_sins = [np.array(datum['sins']) for datum in batch]
    batch_cossins = [np.array(datum['cossins']) for datum in batch]
    batch_colors = [np.array(datum['colors']) for datum in batch]
    batch_highest_contrast_neighbor_colors = [np.array(datum['highest_contrast_neighbor_colors']) for datum in batch]
    batch_invariants = [np.array(datum['invariants']) for datum in batch]
    batch_targets = [datum['target'] for datum in batch]
    batch_img_paths = [datum['img_path'] for datum in batch]

    # convert and pad locations
    batch_locations = torch.nn.utils.rnn.pad_sequence(
        [torch.from_numpy(locations) for locations in batch_locations],
        batch_first=True
        )

    # concatenate orientations, colors and invariants of darker nodes to form features
    batch_feats = []
    for i in range(len(batch)):
        angles = torch.from_numpy(batch_orientations[i]).unsqueeze(1)
        sin = torch.from_numpy(batch_sins[i]).unsqueeze(1)
        cos = torch.from_numpy(batch_cossins[i]).unsqueeze(1)
        colors = torch.from_numpy(batch_colors[i]).unsqueeze(1)
        highest_contrast_neighbor_colors = torch.from_numpy(batch_highest_contrast_neighbor_colors[i]).unsqueeze(1)
        invariants = torch.from_numpy(np.array(batch_invariants[i]))
        feats = torch.cat((sin, cos, colors, highest_contrast_neighbor_colors, invariants), dim=-1)
        batch_feats.append(feats)

    # pad features
    batch_feats = torch.nn.utils.rnn.pad_sequence(batch_feats, batch_first=True).to(torch.float32)

    # obtain metadata and edges
    batch_size = batch_feats.size(0)
    n_nodes = batch_feats.size(1)
    feature_size = batch_feats.size(2)

    edges, _ = get_edges_batch(n_nodes, batch_size)

    return dict(
                locations = (batch_locations.view(-1,2) / 28).to(torch.float32),
                # locations = (batch_locations.view(-1,2)).to(torch.float32),
                feats = batch_feats.view(-1, feature_size).to(torch.float32),
                edges = edges,
                batch_size=batch_size,
                n_nodes=n_nodes,
                feature_size = feature_size,
                img_paths=batch_img_paths,
                target=torch.LongTensor([b['target'] for b in batch]),
        )


class GraphMNIST(torch.utils.data.Dataset):
    def __init__(self, data_root='./graph_mnist/',is_train=True):
        self.data_root = data_root
        self.is_train = is_train
        if self.is_train:
            with open(data_root + 'train_data.pickle', "rb") as fp:
                self.data_dict = pickle.load(fp)
        elif not self.is_train:
            with open(data_root + 'test_data.pickle', "rb") as fp:
                self.data_dict = pickle.load(fp)

        self.data = pd.DataFrame(self.data_dict)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, ix):
        sample = self.data.iloc[ix]
        # to prevent duplicates, mask out brighter partner of superpixel pair
        mask = sample['is_darker']

        locations = []
        orientations = []
        sins, cossins = [], []
        colors = []
        highest_contrast_neighbor_colors = []
        invariants = []

        for i in range(len(mask)):
            if mask[i] == 1:
                locations.append(sample['center_between'][i])
                orientations.append(sample['angle_between'][i])
                sins.append(sample['sin'][i])
                cossins.append(sample['cos'][i])
                colors.append(sample['color'][i])
                highest_contrast_neighbor_colors.append(sample['highest_contrast_neighbor_color'][i])
                invariants.append(sample['invariants'][i])

        target = sample['targets']


        return dict(
            locations=locations,
            orientations=orientations,
            sins=sins,
            cossins = cossins,
            invariants=invariants,
            colors=colors,
            highest_contrast_neighbor_colors=highest_contrast_neighbor_colors,
            target=target,
            img_path=sample['img_paths'],
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

    graph_trainloader = torch.utils.data.DataLoader(graph_mnist_train, batch_size=1, shuffle=True, collate_fn=collate_graph)
    graph_testloader = torch.utils.data.DataLoader(graph_mnist_test, batch_size=1, shuffle=False, collate_fn=collate_graph)


    graph_sample = next(iter(graph_trainloader))
    print(graph_sample.keys())

    edges = [graph_sample['edges'][0].flatten(),graph_sample['edges'][1].flatten()]


    # print(graph_sample['img_paths'][0])
    # print(graph_sample['target'][0])

    # img = np.load(graph_sample['img_paths'][0])
    # plt.imshow(img)

    # coords = graph_sample['locations'][:graph_sample['n_nodes']] * 28
    # print(coords.shape)
    # print(coords)
    # plt.scatter(coords[:,1], coords[:,0])
    # print(graph_sample['feats'].min())

    # print(graph_sample['batch_size'])
    # print()


    # plt.show()
    # print(graph_sample['feature_size'])
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
    

    model = EGNN(
        in_node_nf=1, # 9
        hidden_nf=100,
        out_node_nf=10,
        in_edge_nf=0,
        attention=True,
        normalize=True,
        n_layers=6,
    ).to('cpu')
    # print(model)
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