'''
    Funzioni per generare il Dataset della rete
'''
import tensorflow as tf
import glob
import numpy as np
from itertools import islice

#----------------------------------------------
def check():
    """
    Questa funzione serve per chiedere conferma se quanto inserito è un path per piu di un file o una stringa
    che porta ad un unico file
    """
    while True:
        risposta = input("La stringa inserita è un path per file singolo o per diversi file?(sing/multi): ")
        if risposta == 'multi' or risposta=='sing':
            return risposta
        else:
            print('Deve essere o "multi" o "sing"')


#----------------------------------------------
def numAtomxframe(File):
    """
    numAtomxframe trova tutti i numeri degli atomi per i vari frame/simulazioni

    :File : stringa, path del file da analizzare (dal quale estrarre i dati per training e validation)

    :retunr lines: array di stringhe, contenente tutte le righe del file dati
    :return natom: 1D numpy array,  gli elementi sono int del numero di atomi per frame
    :return equal: variabile bool
    """
    natom = []
    equal = False  #variabile booleana che serve ad affermare se in tutti i frame ho lo stesso numero di atomi 
    ind_n = 3      #riga dove si legge il numero di atomi nella simulazione
    num_comm = 9   #numero di righe di commento che separano i frame (costante in LAMMPS)


    with open(File) as f_input:
        lines = f_input.readlines()
    num_righe = len(lines) #numero di righe del codice
    
    i = 0
    while (i < num_righe):
        n = lines[i+ind_n]
        natom.append(n)
        i = i + int(n) + num_comm
    
    natom = np.array(natom, dtype='i')
    all_zeros = not np.any(natom - natom[0])
    if all_zeros:
        equal = True

    return lines, natom, equal




#https://stackoverflow.com/questions/66549310/concatenate-different-sized-ndarrays





#----------------------------------------------
def Organizza_Data(lines, natom):
    """
    Organizza_Data sistema i dati nella modalita desiderata 

    :lines: array di stringhe, contenente tutte le righe del file dati
    :natom: 1D numpy array,  gli elementi sono int del numero di atomi per frame   

    :return DATA_SETxyz: np array shape=(numeroFrame, numeroAtomi, 3), array delle coordinate dei grafi
    :return DATA_SEThotvec: np array shape=(numeroFrame, numeroAtomi, 2), array degli hot vector per O ed H
    :return Output: np array shape=(numeroFrame, numeroAtomi, 3), array delle Forze
    """
    n_frame = len(natom)
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
    return DATA_SETxyz, DATA_SEThotvec, Output 




#----------------------------------------------
def Data(File):
    """
    DEVO ANCORA COMMMENTARLA BENE
    Data prende un file di lammps restituisce i Data ed i set (da separare) per il training e validation
    
    :File : stringa, path del file da analizzare (dal quale estrarre i dati per training e validation)
    
    :return DATA_SETxyz: np array shape=(numeroFrame, numeroAtomi, 3), array delle coordinate dei grafi
    :return DATA_SEThotvec: np array shape=(numeroFrame, numeroAtomi, 2), array degli hot vector per O ed H
    :return Output: np array shape=(numeroFrame, numeroAtomi, 3), array delle Forze
    
    """
    Input = check()

    if Input == 'sing':
        lines, natom, equal = numAtomxframe(File=File)

        if equal:
            DATA_SETxyz, DATA_SEThotvec, Output = Organizza_Data(lines, natom)
            DATA_SET = tf.ragged.constant([DATA_SETxyz,DATA_SEThotvec]) #in questo oggetto il primo indice con 0 selezione le xyz con 1 gli hotvec
            return DATA_SET, Output
        else:
            ''' E un errore avere il numero diverso, spiega!'''
            print('Nel file sono contenuti frame con numeri di particelle diversi!')
    else:
        Files = glob.glob(File)
        for (file,i) in zip(Files,range(len(Files))):
            lines, natom, equal = numAtomxframe(File=file)

            if equal:
                DATA_SETxyz, DATA_SEThotvec, Output = Organizza_Data(lines, natom)
                DATA_SET = tf.ragged.constant([DATA_SETxyz,DATA_SEThotvec])
                
                #Devo aggiungere le strutture dati luna con laltra per i vari file
                #https://stackoverflow.com/questions/73038417/concatenate-tensors-with-different-shapes-in-tensorflow
                if i == 0:
                    rag_output = tf.RaggedTensor.from_tensor(Output)
                    rag_DATA_SET = tf.RaggedTensor.from_tensor(DATA_SET)
                    DATA_TOT = rag_DATA_SET
                    Outputt_TOT = rag_output
                else:
                    rag_output = tf.RaggedTensor.from_tensor(Output)
                    rag_DATA_SET = tf.RaggedTensor.from_tensor(DATA_SET)
                    DATA_TOT = tf.concat([DATA_TOT,rag_DATA_SET], axis=0)
                    Outputt_TOT = tf.concat([Outputt_TOT,rag_output], axis=0)
            else:
                ''' E un errore avere il numero diverso, spiega!'''
                print('Nel file sono contenuti frame con numeri di particelle diversi!')
        
        return DATA_TOT,Outputt_TOT





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

