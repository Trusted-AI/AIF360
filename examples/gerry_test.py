import warnings
warnings.filterwarnings("ignore")
from aif360.algorithms.inprocessing.gerryfair_classifier import *
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import *


# load data set
data_set = load_preproc_data_adult(sub_samp=1000, balance=True)
max_iterations = 10
C = 100
print_flag = True
gamma = .005


fair_model = GerryFairClassifier(C=C, printflag=print_flag, gamma=gamma, fairness_def='FP',
             max_iters=max_iterations, heatmapflag=False)

# fit method
fair_model = fair_model.fit(data_set, early_termination=True)

# predict method. If threshold in (0, 1) produces binary predictions

dataset_yhat = fair_model.predict(data_set, threshold=False)
