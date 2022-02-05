import torch
from torch_geometric.data import InMemoryDataset, download_url
import pickle
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class GraphMNIST(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    # @property
    # def raw_file_names(self):
    #     return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['graph_mnist.pt']

    # def download(self):
    #     # Download to `self.raw_dir`.
    #     download_url(url, self.raw_dir)
    #     ...

    def get_edges(self, n_nodes):
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    rows.append(i)
                    cols.append(j)

        edges = [rows, cols]

        return edges

    def pre_process(self, mask, loc, feat, target):

        locs = []
        feats = []
        for i in range(len(mask)):
            if mask[i] == 1:
                locs.append(loc[i])
                feats.append(feat[i])

        x = torch.cat((
            torch.tensor(np.array(locs)),
            torch.tensor(np.array(feats)).unsqueeze(1),
            ), dim=1).float()

        n_nodes = x.size(0)

        edges = torch.tensor(np.array(self.get_edges(n_nodes)))
        datum = Data(x, edges, y=torch.tensor(target))


        return datum

    def process(self):
        # load pre-processed mnist graph data (created with SLIC, see generate_graph_mnist.py)
        with open('test_data.pickle', "rb") as fp:
            data_dict = pickle.load(fp)
        
        # Read data into huge `Data` list.
        data_list = [self.pre_process(t[0], t[1], t[2], t[3]) for t in zip(
            data_dict['is_darker'],
            data_dict['center_between'],
            data_dict['highest_contrast_neighbor_color'],
            data_dict['targets'],
            )]


        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])




if __name__ == '__main__':
    dataset = GraphMNIST('.')

    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    datum = next(iter(train_loader))
    print(datum.keys)





