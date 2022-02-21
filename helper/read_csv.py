import csv
import numpy as np

def read_data(csv_file_name):
    F_data = []
    S_data = []
    V_data = []
    D_data = []
    W_data = []
    H_data = []
    with open(csv_file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        
        for i,row in enumerate(csv_reader):
            if i==0:
                continue
            else:
                
                F_data.append(float(row[0]))
                S_data.append(float(row[1]))
                V_data.append(float(row[2]))
                D_data.append(float(row[3]))
                W_data.append(float(row[4]))
                H_data.append(float(row[5]))
    return np.array(F_data),np.array(S_data),np.array(V_data),np.array(D_data),np.array(W_data),np.array(H_data)