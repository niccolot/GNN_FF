'''
    CHANGE THE STRUCTURE OF LAMMPS OUTPUT INTO A GNN DATASET
'''
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
import numpy as np
import os 
from tqdm import tqdm





# CLASSES
#-----------------------------------------------
class LAMMPSDataset(Dataset):
    def __init__(self, root,  test = False, transform=None, pre_transform=None):
        r"""
        root = Where the dataset should be stored. This folder is split
               into raw_dir (raw dataset) and processed_dir (processed dataset)

               /data_
                     |- /raw
                     |- /processed
        """
        #self.filename = filename
        self.test = test
        super(LAMMPSDataset, self).__init__(root, transform, pre_transform)

    @property
    def processed_file_names(self):
        r""" If these files are found in processed_dir, processing is skipped
        """
        return  self.processed_dir+'/tensor_dataset1.pt'

    def process(self):
        r""" Run on all file in raw_dir and process them into data file
            then convert them in Data object and save in proper dataset
        """
        print('Swag')
        i = 0
        for file in os.listdir(self.raw_dir):
            path_file = os.path.join(self.raw_dir, file)
            i += 1
            with open(path_file) as f_input:

                lines = f_input.readlines()
                num_lines = len(lines) #rows number in data file
            f_input.close()
           
            npart = int(lines[3]) #particle number
            boundx = float(lines[5].split()[0]) -float(lines[5].split()[1])
            boundy = float(lines[6].split()[0]) -float(lines[6].split()[1])
            boundz = float(lines[7].split()[0]) -float(lines[7].split()[1])
            bound = np.array([boundx,boundy,boundz]) #buond conditions
            print(bound)

            header_rows = 9
            max_index = int(num_lines/(npart+header_rows))

            self.lst_tensor = []
            for index in tqdm(range(0,max_index)):
                
                DATA_SETxyz, DATA_SEThotvec, Output = rawtoDataL(lines,index,npart)
                data = Data(x =DATA_SETxyz ,
                            y =Output ,
                            z =DATA_SEThotvec ,
                            bounds = bound 
                            )

                self.lst_tensor.append(data)
            
            torch.save(self.lst_tensor,os.path.join(self.processed_dir,f'tensor_dataset{i}.pt'))
            # Because saving a huge python list is rather slow, it is possible to collate the list into one huge Data object via torch_geometric.data.InMemoryDataset.collate() before saving . 
            #Dat, slices = torch_geometric.data.InMemoryDataset.collate(lst_tensor)
            #torch.save((Dat, slices),os.path.join(self.processed_dir,f'tensor_dataset{i}.pt'))


    def len(self):
        return len(self.lst_tensor)


    def get(self,fi):
        r"""
        Equivalent to __getitem__ in pytorch
        """
        data = torch.load(os.path.join(self.processed_dir,
                                           f'tensor_dataset{fi}.pt'))
        return data


def rawtoDataL(lines: list, frame: int,npart: int):
    r"""
    rawtoDataL access an open file and select lines for a fixed frame

    Parameters
    ----------
        - intput (list): opened file
        - frame (int): frame
        - npart (int): number of particle in the frame

        
    Returns
    ---------
        - DATA_SETxyz (tensor):  size=(numeroAtomi, 3), coordinates tensor
        - DATA_SEThotvec (tensor): size=( numeroAtomi, 1),  hot vector tensor for O and H
        - Output (tensor): size=( numeroAtomi, 3), Force tensor    
    """
    
    DATA_SETxyz = np.zeros(shape=(npart, 3), dtype='f')
    DATA_SEThotvec = np.zeros(npart, dtype='i')
    Output = np.zeros(shape=(npart, 3), dtype='f')
    header_rows = 9
    start_row = frame*(npart+header_rows)+header_rows
    end_row = start_row + npart



    for (row,i) in zip(lines[start_row:end_row],range(npart)):

        DATA_SETxyz[i][0] = float(row.split()[1])
        DATA_SETxyz[i][1] = float(row.split()[2])
        DATA_SETxyz[i][2] = float(row.split()[3])
        Output[i][0] = float(row.split()[4])
        Output[i][0] = float(row.split()[5])
        Output[i][0] = float(row.split()[6])
        
        DATA_SEThotvec[i] = int(float(row.split()[0]))

    DATA_SETxyz = torch.from_numpy(DATA_SETxyz)
    DATA_SEThotvec = torch.from_numpy(DATA_SEThotvec)
    Output = torch.from_numpy(Output)


    return DATA_SETxyz, DATA_SEThotvec, Output


#-----------------------------------------------
if __name__=='__main__':
    DATA_PATH= '/home/castelli/Documents/GNNFF/Progetto_Esame2/data'



    dataset = LAMMPSDataset(root=DATA_PATH)

    swagData = dataset.get(1)[0]
    print(swagData.x[:1])
    print(swagData.z[:1])
    print(swagData.bounds)
    print(len(dataset))
