# testing integration of gerry fair with aif360
import sys
sys.path.append("../")
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
from aif360.algorithms.inprocessing.gerryfair_classifier import *




dataset_orig = load_preproc_data_adult()

C = 10
printflag = True
gamma = .01
max_iters = 10
fair_def = 'FP'

fair_model = Model(C=C, printflag=printflag, gamma=gamma, fairness_def=fair_def)
fair_model.set_options(max_iters=max_iters)


# fit method
communities_all_errors, communities_violations = fair_model.fit(dataset_orig,
                                                                early_termination=True, return_values=True)
# predict method
dataset_yhat = fair_model.predict(dataset_orig)

# fit_transform method will throw an error (not implemented)
fair_model.fit_transform(dataset_orig)

# test other methods

# save heatmap
fair_model.save_heatmappython (fair_model.max_iters, dataset_orig, dataset_yhat.labels, None, None, force_heatmap=True)

# run & create pareto curves
gamma_list = [.01, .02, .03, 1.0]
fair_model.pareto(dataset_orig, gamma_list)

# auditing a classifier for unfairness
# instantiate auditor
dataset_orig = load_preproc_data_adult()
auditor = Auditor(dataset_orig, 'FP')
auditor.get_group(dataset_yhat.labels, auditor.get_baseline(dataset_orig.labels, dataset_yhat.labels))


