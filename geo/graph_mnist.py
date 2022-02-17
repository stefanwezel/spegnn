import torch
from torch_geometric.data import InMemoryDataset, download_url
import pickle
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class GraphMNIST(InMemoryDataset):
    def __init__(self, root, is_train=True,transform=None, pre_transform=None, pre_filter=None):
        self.is_train = is_train

        super().__init__(root, transform, pre_transform, pre_filter)

        if self.is_train:
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.data, self.slices = torch.load(self.processed_paths[1])
    # @property
    # def raw_file_names(self):
    #     return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['graph_mnist_train.pt', 'graph_mnist_test.pt',]

    # @property
    # def is_train(self, is_train):
        # return True
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


    def preprocess_datum(self, datum_dict):

        mask = datum_dict['mask']
        loc = datum_dict['loc']
        sin = datum_dict['sin']
        cos = datum_dict['cos']
        color = datum_dict['color']
        invariant = datum_dict['invariant']
        target = datum_dict['target']

        locs = []
        sins = []
        coss = []
        colors = []
        invariants = []

        for i in range(len(mask)):
            if mask[i] == 1:
                locs.append(loc[i])
                sins.append(sin[i])
                coss.append(cos[i])
                colors.append(color[i])
                invariants.append(invariant[i])

        x = torch.cat((
            torch.tensor(np.array(locs)/28),
            torch.tensor(np.array(sins)).unsqueeze(1),
            torch.tensor(np.array(coss)).unsqueeze(1),
            torch.tensor(np.array(colors)/255).unsqueeze(1),
            torch.tensor(np.array(invariants)),
            ), dim=1).float()

        n_nodes = x.size(0)

        edges = torch.tensor(np.array(self.get_edges(n_nodes)))
        datum = Data(x, edges, y=torch.tensor(target))

        return datum



    def process(self):
        # load pre-processed mnist graph data (created with SLIC, see generate_graph_mnist.py)        
        if self.is_train:
            with open('train_data.pickle', "rb") as fp:
                data_dict = pickle.load(fp)
        else:
            with open('test_data.pickle', "rb") as fp:
                data_dict = pickle.load(fp)
        
        # Read data into huge `Data` list.
        data_list = [self.preprocess_datum(
                dict(
                    mask=t[0],
                    loc=t[1],
                    sin=t[2],
                    cos=t[3],
                    color=t[4],
                    invariant=t[5],
                    #
                    target=t[-1],
                    )
                ) for t in zip(
            data_dict['is_darker'],
            data_dict['center_between'],
            data_dict['sin'],
            data_dict['cos'],
            data_dict['highest_contrast_neighbor_color'],
            data_dict['invariants'],
            # 
            data_dict['targets'],
            )]


        data, slices = self.collate(data_list)


        if self.is_train:
            torch.save((data, slices), self.processed_paths[0])
        else:
            torch.save((data, slices), self.processed_paths[1])



if __name__ == '__main__':
    train_dataset = GraphMNIST('.')
    test_dataset = GraphMNIST('.', is_train=False)
    print(train_dataset)
    print(test_dataset)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    datum = next(iter(train_loader))
    print(datum.y)





