
# output: the predicted drug sensitivity of our DIVERSE model

import numpy
import csv
from source.hmf_Gibbs import HMF_Gibbs
from source.mask import zero_indices

''' Model settings '''

#########################################

#matrix factorization settings
iterations, burn_in, thinning = 400, 350, 2

settings = {
    'priorF'  : 'exponential',
    'priorG'  : 'exponential',
    'priorSn' : 'exponential',
    'priorSm' : 'exponential',
    'orderF'  : 'columns',
    'orderG'  : 'columns',
    'orderSn' : 'individual',
    'orderSm' : 'individual',
    'ARD'     : True
}


hyperparameters = {
    'alphatau' : 1.,
    'betatau'  : 1.,
    'alpha0'   : 0.001,
    'beta0'    : 0.001,
    'lambdaF'  : 0.1,
    'lambdaG'  : 0.1,
    'lambdaSn' : 0.1,
    'lambdaSm' : 0.1,
}

init = {
    'F'       : 'random',
    'Sn'      : ['least','least','least'],
    'Sm'      : 'least',
    'G'       : 'least',
    'lambdat' : 'exp',
    'tau'     : 'exp'
}


''' Load in data '''
# main data (drug response)
dataset = numpy.genfromtxt("data/scaled_pd.txt",dtype=float,usemask=True,missing_values=numpy.nan)
R_main = dataset.data
R_main[numpy.isnan(R_main)] = 0
R_main = numpy.transpose(R_main)

#similarity data
C = numpy.genfromtxt("data/scaled_dd.txt",dtype=float,usemask=True,missing_values=numpy.nan)
C_1 = C.data
C_1[numpy.isnan(C_1)] = 0.

#feature data
D1 = numpy.genfromtxt("data/scaled_gd.txt",dtype=float,usemask=True,missing_values=numpy.nan)
D_1 = D1.data
D_1[numpy.isnan(D_1)] = 0.

D2 = numpy.genfromtxt("data/scaled_gg.txt",dtype=float,usemask=True,missing_values=numpy.nan)
D_2 = D2.data
D_2[numpy.isnan(D_2)] = 0.

D3 = numpy.genfromtxt("data/scaled_pg.txt",dtype=float,usemask=True,missing_values=numpy.nan)
D_3= D3.data 
D_3[numpy.isnan(D_3)] = 0.


alpha_PD = 1
#alpha's are importance weights for all side data sets
alpha = [0.2, 0.4, 0.6, 0.8, 1.0]


M_Y_training_folds = []
M_Y_test_folds = []

#read training and test data sets
for t in range(5):
    with open("folds_out/folds_1/folds_train_exp_{}_1.csv".format(t), "r+") as f_train:
        reader_train = csv.reader(f_train, delimiter=",")
        reader_train = numpy.array([numpy.array(xi) for xi in reader_train], dtype=float)
        M_Y_training_folds.append(reader_train)

    with open("folds_out/folds_1/folds_test_exp_{}_1.csv".format(t), "r+") as f_test:
        reader_test = csv.reader(f_test, delimiter=",")
        reader_test = numpy.array([numpy.array(xi) for xi in reader_test], dtype=float)
        M_Y_test_folds.append(reader_test)


#Use a method to run the cross-validation under different settings
def run_all_settings(alpha):

    MSE_list_DD = []
    MSE_list_PG = []
    MSE_list_GG = []

    n_folds = 5
    K = {'drugs':40, 'cell_lines':50, 'genes':30}


    #M's are mask matrices
    Mm_1 = numpy.zeros(C_1.shape)
    for i in range(C_1.shape[0]):
        for j in range(C_1.shape[1]):
            if C_1[i,j] != 0:
                Mm_1[i,j] = 1

    Ml_1 = numpy.zeros(D_1.shape)
    for i in range(D_1.shape[0]):
        for j in range(D_1.shape[1]):
            if D_1[i,j] != 0:
                Ml_1[i,j] = 1

    Ml_2 = numpy.zeros(D_2.shape)
    for i in range(D_2.shape[0]):
        for j in range(D_2.shape[1]):
            if D_2[i,j] != 0:
                Ml_2[i,j] = 1

    Ml_3 = numpy.zeros(D_3.shape)
    for i in range(D_3.shape[0]):
        for j in range(D_2.shape[1]):
            if D_3[i,j] != 0:
                Ml_3[i,j] = 1


    #zero indices in main matrix
    zero_index = zero_indices(R_main)

    M_train_folds = []
    M_test_folds = []

    # assign 0 for empty indices for training and test sets
    for i in range(5):
        for k in range(len(zero_index)):
            a = zero_index[k][0]
            b = zero_index[k][1]
            M_Y_training_folds[i][a,b] = 0
            M_Y_test_folds[i][a,b] = 0
        M_train_folds.append(M_Y_training_folds[i])
        M_test_folds.append(M_Y_test_folds[i])


############################################################################33
#####PD + DD variations (integrate drug similarity data)

    for j in range(5):
        ''' Run HMF to predict Y from X '''

        all_MSE = numpy.zeros(n_folds)
        R_pred = []

        for i in range(5):
            M_train = M_train_folds[i]
            M_test = M_test_folds[i]

            R = [
                 (R_main,   M_train,   'drugs',      'cell_lines', alpha_PD)
                 ]
            C = [
                 (C_1,      Mm_1,      'drugs',        alpha[j])
                 ]
            D = [] 

            
            ''' Train and predict '''
            HMF = HMF_Gibbs(R,C,D,K,settings,hyperparameters)
            HMF.initialise(init)
            HMF.run(iterations)
            
            ''' Compute the performances '''
            performances = HMF.predict_Rn(n=0,M_pred=M_test,burn_in=burn_in,thinning=thinning)

            #add all predicted data into R_pred
            predicted_R = HMF.return_Rn(n=0,burn_in=burn_in,thinning=thinning)
            R_pred.append(predicted_R)

            with open("predicted_data_out/folds_1_predicted/dd/R_pred_dd_par_{}_fold_{}.csv".format(j,i), "w") as f:
                writer = csv.writer(f, delimiter=",")
                writer.writerows(predicted_R)
            
            all_MSE[i] = performances['MSE']

        MSE_list_DD.append(all_MSE.mean())

    #find the min mse and its index between importance weights
    index_min_MSE_DD = MSE_list_DD.index(min(MSE_list_DD))

#######################################################################################################################
#####PD + DD + PG variations (integrate drug similarity and gene expression data)

    for j in range(5):
        ''' Run HMF to predict Y from X '''
        all_MSE = numpy.zeros(n_folds)
        R_pred = []

        for i in range(5):
            M_train = M_train_folds[i]
            M_test = M_test_folds[i]


            R = [
                 (R_main,   M_train,   'drugs',      'cell_lines', alpha_PD),
                 (D_3,      Ml_3,      'cell_lines', 'genes', alpha[j])
                 ]
            C = [
                 (C_1,      Mm_1,      'drugs',        alpha[index_min_MSE_DD])
                 ]
            D = [] 

            
            ''' Train and predict '''
            HMF = HMF_Gibbs(R,C,D,K,settings,hyperparameters)
            HMF.initialise(init)
            HMF.run(iterations)

            ''' Compute the performances '''
            performances = HMF.predict_Rn(n=0,M_pred=M_test,burn_in=burn_in,thinning=thinning)

            #add all predicted data into R_pred
            predicted_R = HMF.return_Rn(n=0,burn_in=burn_in,thinning=thinning)
            R_pred.append(predicted_R)

            with open("predicted_data_out/folds_1_predicted/pg/R_pred_pg_par_{}_fold_{}.csv".format(j,i), "w") as f:
                writer = csv.writer(f, delimiter=",")
                writer.writerows(predicted_R)

            all_MSE[i] = performances['MSE']

        MSE_list_PG.append(all_MSE.mean())


    #find the min mse between difference importance weights
    index_min_MSE_PG = MSE_list_PG.index(min(MSE_list_PG))

#######################################################################################################################
#####PD + DD + PG + GG variations (integrate drug similarity, gene expression and gene-gene interaction data)

    for j in range(5):
        ''' Run HMF to predict Y from X '''
        all_MSE = numpy.zeros(n_folds)
        R_pred = []

        for i in range(5):
            M_train = M_train_folds[i]
            M_test = M_test_folds[i]

 
            R = [
                 (R_main,   M_train,   'drugs',      'cell_lines', alpha_PD),
                 (D_3,      Ml_3,      'cell_lines', 'genes', alpha[index_min_MSE_PG])
                 ]
            C = [
                 (C_1,      Mm_1,      'drugs',        alpha[index_min_MSE_DD]),
                 (D_2,      Ml_2,      'genes',        alpha[j])
                 ]
            D = [] 

            
            ''' Train and predict '''
            HMF = HMF_Gibbs(R,C,D,K,settings,hyperparameters)
            HMF.initialise(init)
            HMF.run(iterations)

            ''' Compute the performances '''
            performances = HMF.predict_Rn(n=0,M_pred=M_test,burn_in=burn_in,thinning=thinning)

            # add all predicted data into R_pred
            predicted_R = HMF.return_Rn(n=0, burn_in=burn_in, thinning=thinning)
            R_pred.append(predicted_R)

            with open("predicted_data_out/folds_1_predicted/gg/R_pred_gg_par_{}_fold_{}.csv".format(j,i), "w") as f:
                writer = csv.writer(f, delimiter=",")
                writer.writerows(predicted_R)

            all_MSE[i] = performances['MSE']

        MSE_list_GG.append(all_MSE.mean())


    #find the min mse between difference importance weights
    index_min_MSE_GG = MSE_list_GG.index(min(MSE_list_GG))

#######################################################################################################################
#PD + DD + PG + GG + GD variations (integrate drug similarity, gene expression, gene-gene interaction  and drug-target data)
#use matrix-tri-factorization for drug-target data

    for j in range(5):
        ''' Run HMF to predict Y from X '''
        R_pred = []

        for i in range(5):
            M_train = M_train_folds[i]


            R = [
                 (R_main,   M_train,   'drugs',      'cell_lines', alpha_PD),
                 (D_3,      Ml_3,      'cell_lines', 'genes', alpha[index_min_MSE_PG]),
                 (D_1,      Ml_1,      'genes',      'drugs', alpha[j])
                 ]
            C = [
                 (C_1,      Mm_1,      'drugs',        alpha[index_min_MSE_DD]),
                 (D_2,      Ml_2,      'genes',        alpha[index_min_MSE_GG])
                 ]
            D = [] 

            
            ''' Train and predict '''
            HMF = HMF_Gibbs(R,C,D,K,settings,hyperparameters)
            HMF.initialise(init)
            HMF.run(iterations)
            predicted_R = HMF.return_Rn(n=0,burn_in=burn_in,thinning=thinning)

            # add all predicted data into R_pred
            R_pred.append(predicted_R)

            with open("predicted_data_out/folds_1_predicted/gd_mtf/R_pred_gd_par_{}_fold_{}.csv".format(j,i), "w") as f:
                writer = csv.writer(f, delimiter=",")
                writer.writerows(predicted_R)

#######################################################################################################################
#PD + DD + PG + GG + GD variations (integrate drug similarity, gene expression, gene-gene interaction  and drug-target data)
# use matrix-bi-factorization for drug-target data

    for j in range(5):
        ''' Run HMF to predict Y from X '''
        R_pred = []

        for i in range(5):

            M_train = M_train_folds[i]


            R = [
                 (R_main,   M_train,   'drugs',      'cell_lines', alpha_PD),
                 (D_3,      Ml_3,      'cell_lines', 'genes', alpha[index_min_MSE_PG])
                 ]
            C = [
                 (C_1,      Mm_1,      'drugs',        alpha[index_min_MSE_DD]),
                 (D_2,      Ml_2,      'genes',        alpha[index_min_MSE_GG])
                 ]
            D = [(D_1,      Ml_1,      'genes',      'drugs', alpha[j])] 

            
            ''' Train and predict '''
            HMF = HMF_Gibbs(R,C,D,K,settings,hyperparameters)
            HMF.initialise(init)
            HMF.run(iterations)
            predicted_R = HMF.return_Rn(n=0,burn_in=burn_in,thinning=thinning)

            # add all predicted data into R_pred
            R_pred.append(predicted_R)

            with open("predicted_data_out/folds_1_predicted/gd_mf/R_pred_gd_par_{}_fold_{}.csv".format(j,i), "w") as f:
                writer = csv.writer(f, delimiter=",")
                writer.writerows(predicted_R)


''' Run all the settings '''
run_all_settings(alpha)













