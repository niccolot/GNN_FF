'''
    CHANGE THE STRUCTURE OF LAMMPS OUTPUT INTO A GNN DATASET
'''
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
import numpy as np
import os #?
from itertools import islice
from tqdm import tqdm
import glob 




# CLASSES
#-----------------------------------------------
class BOXpbcDataset(Dataset):
    def __init__(self, root,  test = False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
               into raw_dir (downloaded dataset) and processed_dir (processed dataset)

               /data_
                     |- /raw_dir
                     |- /processed_dir
        """
        #self.filename = filename
        self.test = test
        super(BOXpbcDataset, self).__init__(root, transform, pre_transform)
        
    
    @property
    def raw_file_names(self):
        """If the file exists in raw_dir, the download is not triggered
           (The download func. is not implemented here)
        """
        return 'Data*'
    
    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped
        """
        return 'not_implemented'
    
    def download(self):
        pass

    def process(self):
        """ Run on all file in raw_dir and process them into data file
            then convert them in Data object and save in proper dataset
        """
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

            header_rows = 9
            max_index = int(num_lines/(npart+header_rows))
            for index in tqdm(range(0,max_index)):
                
                DATA_SETxyz, DATA_SEThotvec, Output = self.rawtoDataL(lines,index,npart)
                data = Data(x =DATA_SETxyz ,
                            y =Output ,
                            z =DATA_SEThotvec ,
                            bounds = bound 
                            )

                if self.test:
                    torch.save(data, 
                        os.path.join(self.processed_dir, 
                                    f'data_test_{i}_{index}.pt'))
                else:
                    torch.save(data, 
                        os.path.join(self.processed_dir, 
                                    f'data_{i}_{index}.pt'))


    def rawtoDataL(self,lines: list, frame: int,npart: int):
        """
        rawtoDataL access an open file and select lines for a fixed frame

        Parameters
        ----------
            - intput (list): opened file
            - frame (int): frame
            - npart (int): number of particle in the frame

        
        Results
        ---------
            - DATA_SETxyz (tensor):  size=(numeroAtomi, 3), coordinates tensor
            - DATA_SEThotvec (tensor): size=( numeroAtomi, 1),  hot vector tensor for O and H
            - Output (tensor): size=( numeroAtomi, 3), Force tensor
        """
        DATA_SETxyz = np.zeros(shape=(npart, 3), dtype='f')
        #DATA_SEThotvec = np.zeros(shape=(npart, 1), dtype='i')
        DATA_SEThotvec = np.zeros(npart, dtype='i')
        Output = np.zeros(shape=(npart, 3), dtype='f')
        header_rows = 9
        start_row = frame*(npart+header_rows)+header_rows
        end_row = start_row + npart
        #print(start_row, end_row)


        for (row,i) in zip(lines[start_row:end_row],range(npart)):
            #print(row,i)
            #print(int(row.split()[0]))
            DATA_SETxyz[i][0] = float(row.split()[1])
            DATA_SETxyz[i][1] = float(row.split()[2])
            DATA_SETxyz[i][2] = float(row.split()[3])
            Output[i][0] = float(row.split()[4])
            Output[i][0] = float(row.split()[5])
            Output[i][0] = float(row.split()[6])
            #DATA_SEThotvec[i] = float(row.split()[0])
            DATA_SEThotvec[i] = int(float(row.split()[0]))

        DATA_SETxyz = torch.from_numpy(DATA_SETxyz)
        DATA_SEThotvec = torch.from_numpy(DATA_SEThotvec)
        Output = torch.from_numpy(Output)

        return DATA_SETxyz, DATA_SEThotvec, Output


    def len(self):
        return self.data.shape[0]
    
    def get(self,fi, idx):
        """
        Equivalent to __getitem__ in pytorch
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir,
                                           f'data_test_{fi}_{idx}.pt'))
        
        else:
            data = torch.load(os.path.join(self.processed_dir,
                                           f'data_{fi}_{idx}.pt'))
        return data


#-----------------------------------------------
if __name__=='__main__':
    DATA_PATH= '/home/Lorenzo/GNNFF/Test/data/'



    dataset = BOXpbcDataset(root=DATA_PATH)

    swagData = dataset.get(1,0)
    print(swagData.x[:1])
    #print(swagData.y[:1])
    #print(swagData.z[:1])
    #print(swagData.bounds)
