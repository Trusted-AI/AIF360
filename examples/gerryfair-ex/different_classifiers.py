from aif360.algorithms.inprocessing.gerryfair_classifier import Model
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
from sklearn import svm
from sklearn import tree
from sklearn.kernel_ridge import KernelRidge
from sklearn import linear_model
import pickle
import matplotlib.pyplot as plt


def single_trial(dataset, predictor='kernel', C=15, printflag=True, gamma=0.1, max_iter=10):
    if predictor == 'kernel':
        model = KernelRidge(alpha=1.0, gamma=1.0, kernel='rbf')
    elif predictor == 'tree':
        model = tree.DecisionTreeRegressor()
    elif predictor == 'svm':
        model = svm.LinearSVR()
    else:
        model = linear_model.LinearRegression()
    fair_clf = Model(C=C, printflag=printflag, gamma=gamma, predictor=model, max_iters=max_iter)
    fair_clf.fit(dataset)


def multiple_comparision(dataset, gamma_list=tuple([0.01]), max_iter_list=tuple([5]), C=15):
    ln_predictor = linear_model.LinearRegression()
    svm_predictor = svm.LinearSVR()
    tree_predictor = tree.DecisionTreeRegressor()
    kernel_predictor = KernelRidge(alpha=1.0, gamma=1.0, kernel='rbf')
    printflag = False
    predictor_dict = {'Linear': ln_predictor,
                      'SVR': svm_predictor,
                      'DT': tree_predictor,
                      'RBF Kernel': kernel_predictor}
    results_dict = {}
    # For each model, train a Pareto curve
    for max_iter in max_iter_list:
        for curr_predictor in predictor_dict.keys():
            print('Curr Predictor: ')
            print(curr_predictor)
            predictor = predictor_dict[curr_predictor]
            fair_clf = Model(C=C, printflag=printflag, gamma=1, predictor=predictor, max_iters=max_iter)
            print(fair_clf.predictor)
            all_errors, all_fp = fair_clf.pareto(dataset, gamma_list)
            results_dict[curr_predictor] = {'Errors' : all_errors, 'FP_disp': all_fp}
    # pickle.dump(results_dict, open('results_max' + str(max_iter_list) + '_gammas' + str(gamma_list) + '.pkl', 'wb'))

    print(results_dict)


def multiple_pareto(dataset, gamma_list=[0.002, 0.005, 0.01, 0.02, 0.05, 0.1], save_results=True):


    ln_predictor = linear_model.LinearRegression()
    svm_predictor = svm.LinearSVR()
    tree_predictor = tree.DecisionTreeRegressor(max_depth=3)
    kernel_predictor = KernelRidge(alpha=1.0, gamma=1.0, kernel='rbf')
    predictor_dict = {'Linear': {'predictor': ln_predictor, 'iters': 100},
                      'SVR': {'predictor': svm_predictor, 'iters': 10},
                      'DT': {'predictor': tree_predictor, 'iters': 100},
                      'DT': {'predictor': kernel_predictor, 'iters': 100}}

    results_dict = {}

    for pred in predictor_dict:
        print('Curr Predictor: {}'.format(pred))
        predictor = predictor_dict[pred]['predictor']
        max_iters = predictor_dict[pred]['iters']
        fair_clf = Model(C=100, printflag=True, gamma=1, predictor=predictor, max_iters=max_iters)
        fair_clf.set_options(max_iters=max_iters)
        errors, fp_violations, fn_violations = fair_clf.pareto(dataset, gamma_list)
        results_dict[pred] = {'errors': errors, 'fp_violations': fp_violations, 'fn_violations': fn_violations}
        plt.plot(errors, fp_violations, label=pred)

    if save_results:
        pickle.dump(results_dict, open('results_dict_' + str(gamma_list) + '_gammas' + str(gamma_list) + '.pkl', 'wb'))

    plt.xlabel('Error')
    plt.ylabel('Unfairness')
    plt.legend()
    plt.title('Error vs. Unfairness\n(Communities & Crime Dataset)')
    plt.show()


if __name__ == '__main__':
    dataset = load_preproc_data_adult()
    multiple_pareto(dataset)
