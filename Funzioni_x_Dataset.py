'''
    Funzioni per generare il Dataset della rete
'''
import numpy as np
from itertools import islice



#----------------------------------------------
def numAtomxframe(File):
    """
    numAtomxframe trova tutti i numeri degli atomi per i vari frame/simulazioni

    :File : stringa path del file da analizzare (dal quale estrarre i dati per training e validation)

    :retunr lines: array of string with al files strings
    :return natom: 1D numpy array  gli elementi sono int del numero di atomi per frame
    :return equal: variabile bool
    """
    natom = []
    equal = False  #variabile booleana che serve ad affermare se in tutti i frame ho lo stesso numero di atomi 
    ind_n = 4      #riga dove si legge il numero di atomi nella simulazione
    num_comm = 9   #numero di righe di commento che separano i frame (costante in LAMMPS)


    with open(File) as f_input:
        lines = f_input.readlines()
    num_righe = len(lines) #numero di righe del codice
    
    i = 0
    while (i < num_righe):
        n = lines[i+ind_n]
        natom.append(n)
        i = i + n + num_comm
    
    natom = np.array(natom, dtype='i')
    all_zeros = not np.any(natom - natom[0])
    if all_zeros:
        equal = True

    return lines, natom, equal







#----------------------------------------------
def Data(File):
    """
    Data prende un file di lammps restituisce i Data ed i set (da separare) per il training e validation
    
    :File : stringa path del file da analizzare (dal quale estrarre i dati per training e validation)
    
    :return DATA_SETxyz: np array shape=(numeroFrame, numeroAtomi, 3) array delle coordinate dei grafi
    :return DATA_SEThotvec: np array shape=(numeroFrame, numeroAtomi, 2) array degli hot vector per O ed H
    :return DATA_SETxyz: np array shape=(numeroFrame, numeroAtomi, 3) array delle coordinate dei grafi
    
    """
    lines, natom, equal = numAtomxframe(File=File)
    n_frame = len(natom)

    

    if equal:
        DATA_SETxyz = np.array(shape=(n_frame, natom[0], 3), dtype='f')
        DATA_SEThotvec = np.array(shape=(n_frame, natom[0], 2), dtype='i')
        Output = np.array(shape=(n_frame, natom[0], 3), dtype='f')
        for i in range(n_frame):
            A = 9+i*(natom[0]+9)
            B = (i+1)*(natom[0]+9)

            j = 0
            for row in lines[A:B]:
                DATA_SETxyz[i][j][0] = float(row.split()[1])
                DATA_SETxyz[i][j][1] = float(row.split()[2])
                DATA_SETxyz[i][j][2] = float(row.split()[3])
                Output[i][j][0] = float(row.split()[4])
                Output[i][j][0] = float(row.split()[5])
                Output[i][j][0] = float(row.split()[6])
                Type = int(row.split()[0])

                if Type == 2: #nella simulazione 2 indica Ossigeno ed 1 indica idrogeno che qui diventano [0,1] e [1,0]
                    DATA_SEThotvec[i][j][0] = 0
                    DATA_SEThotvec[i][j][1] = 1
                else:
                    DATA_SEThotvec[i][j][0] = 1
                    DATA_SEThotvec[i][j][1] = 0
                j+=1

    else:
        pass

    return DATA_SETxyz, DATA_SEThotvec, Output



#----------------------------------------------
def Mix():
    """
    Mix mette in ordine sparso i frame organizzati dalla funzione Data 
    con lo scopo poi di diverdi in Training set e Validation set.

    """


#----------------------------------------------
def Edge(N):
    """
    Edge trova i primi N vicini di un nodo e li collega con degli edge 

    :N : int numero di primi vicini da collegare con i vari vertici
    """