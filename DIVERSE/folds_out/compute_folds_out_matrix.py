


from sklearn.model_selection import KFold

import numpy
import csv
import math



#M's are mask matrices
#R_main, Mn_main, cell_lines, drugs = load_data_without_empty(input_folder+"scaled_pd.txt")

dataset = numpy.genfromtxt("data/scaled_pd.txt",dtype=float,usemask=True,missing_values=numpy.nan)
R= dataset.data
R[numpy.isnan(R)] = 0.
    
R_main  = R


    # list of training and test folds
M_Y_training_folds = []
M_Y_test_folds = []
M_Y_training_index = []
M_Y_test_index = []


kf = KFold(n_splits=5, shuffle=True)

data_dir = "/m/home/home7/72/guvencb1/unix/Desktop/DRNMTF"

for train_index, test_index in kf.split(R_main):
    y_train, y_test = R_main[train_index], R_main[test_index]
    M_Y_train = numpy.ones(R_main.shape)
    M_Y_train[test_index] = 0.
    M_Y_test = 1. - M_Y_train
    M_Y_training_folds.append(M_Y_train)
    M_Y_test_folds.append(M_Y_test)
    M_Y_training_index.append(train_index)
    M_Y_test_index.append(test_index)



for i, folds in enumerate(M_Y_training_folds):
    with open("{}/folds_train_exp_{}_1.csv".format(data_dir,i), "wb") as f:
       writer = csv.writer(f, delimiter=",")
       writer.writerows(folds)


for i, folds in enumerate(M_Y_test_folds):
    with open("{}/folds_test_exp_{}_1.csv".format(data_dir,i), "wb") as f:
       writer = csv.writer(f, delimiter=",")
       writer.writerows(folds)



for i, folds in enumerate(M_Y_training_index):
    with open("{}/folds_train_index_{}_1.csv".format(data_dir,i), "wb") as f:
       writer = csv.writer(f, delimiter=",")
       for row in folds:
           writer.writerow([row])


for i, folds in enumerate(M_Y_test_index):
    with open("{}/folds_test_index_{}_1.csv".format(data_dir,i), "wb") as f:
       writer = csv.writer(f, delimiter=",")
       for row in folds:
           writer.writerow([row])








