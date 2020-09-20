import numpy
import csv
from scipy import stats
import itertools

K = 5
M_Y_test_folds = []
folds_test_index = []

#read all test data and indices
for t in range(5):
    with open("folds_out/folds_1/folds_test_exp_{}_1.csv".format(t), "r+") as f_test:
        reader_test = csv.reader(f_test, delimiter=",")
        reader_test = numpy.array([numpy.array(xi) for xi in reader_test], dtype=float)
        M_Y_test_folds.append(reader_test)

    with open("folds_out/folds_1/folds_test_index_{}_1.csv".format(t), "r+") as folds_test_index:
        reader_test_index = csv.reader(folds_test_index, delimiter=",")
        reader_test_index = numpy.array([numpy.array(xi) for xi in reader_test_index], dtype=float)
        folds_test_index.append(reader_test_index)


def zero_indices(M):
    (I,J) = numpy.array(M).shape
    return [(i,j) for i,j in itertools.product(range(I),range(J)) if M[i][j] == 0]

#import data
patient_drug = numpy.genfromtxt("scaled_pd.txt", dtype = float, usemask = True, missing_values = numpy.nan)
patient_drug = patient_drug.data
patient_drug[numpy.isnan(patient_drug)] = 0.
drug_patient = numpy.transpose(patient_drug)


zero_index = zero_indices(drug_patient)

for i in range(5):
    for k in range(len(zero_index)):
        a = zero_index[k][0]
        b = zero_index[k][1]
        M_Y_test_folds[i][a,b] = 0

#read predicted (approximated) data
def read_folds_predicted():
    predicted_data = []
    for j in range(K):
        folds = numpy.genfromtxt("predicted_data/folds_1_predicted/R_pred_dd_fold_{}.csv".format(j), delimiter=",")
        predicted_data.append(folds)
    return predicted_data
pred_R = read_folds_predicted()


''' Functions for computing MSE, drug averaged Sc (Spearman correlation) '''

''' Return the MSE of predictions in R_pred, expected values in R, for the entries in M. '''
def compute_MSE(M,R,R_pred):
    return (M * (R-R_pred)**2).sum() / float(M.sum())

''' Return the drug averaged Spearman coorleation scores of predictions in R_pred, expected values in R, for the entries in M. '''
def compute_Rp_drug_Sc(M_index,M,R,R_pred):
    spearman_coef = []
    for i in range(len(M_index)):
        a = int(M_index[i])
        observed_R = M * R
        predicted_R = M * R_pred
        spearman_drug = stats.spearmanr(observed_R[a,:], predicted_R[a,:])
        spearman_coef.append(spearman_drug[0])
    return numpy.mean(spearman_coef)


'''
##drug-averaged Spearman evaluation for in-matrix prediction results
def compute_Rp_drug_Sc_in_matrix(M,R,R_pred):
    spearman_coef = []
    for i in range(len(R)):
        observed_R = M * R
        predicted_R = M * R_pred
        spearman_drug =stats.pearsonr(observed_R[i,:], predicted_R[i,:])
        spearman_coef.append(spearman_drug[0])
    return numpy.mean(spearman_coef)
'''


all_MSE = []
all_Sc_drug = []
for i in range(K):
    test_index = folds_test_index[i]
    test_M = M_Y_test_folds[i]

    R = pred_R[i]

    mse = compute_MSE(test_M,drug_patient,R)
    all_MSE.append(mse)

    sc_drug = compute_Rp_drug_Sc(test_index,test_M,drug_patient,R)
    all_Sc_drug.append(sc_drug)

MSE = numpy.array(all_MSE)
Sc_drug = numpy.array(all_Sc_drug)

print ("mse:", MSE.mean(), "std:",MSE.std())
print ("Sc drug:", Sc_drug.mean(), "std:",Sc_drug.std())









