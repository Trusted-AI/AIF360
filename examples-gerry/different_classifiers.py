import gerryfair
from gerryfair.model import Model
from gerryfair.clean import clean_dataset
from gerryfair.oracles import TorchPredictor, ThreeLayerNet

from sklearn import svm
from sklearn import tree
from sklearn.kernel_ridge import KernelRidge
from sklearn import linear_model

import pickle
import matplotlib.pyplot as plt

dataset = "./dataset/communities.csv"
attributes = "./dataset/communities_protected.csv"
centered = True
X, X_prime, y = clean_dataset(dataset, attributes, centered)

def single_trial():
    ln_predictor = linear_model.LinearRegression()
    svm_predictor = svm.LinearSVR()
    tree_predictor = tree.DecisionTreeRegressor()
    kernel_predictor = KernelRidge(alpha=1.0, gamma=1.0, kernel='rbf')
    C = 15
    printflag = True
    gamma = 0.1
    max_iter = 10
    fair_clf = Model(C=C, printflag=printflag, gamma=gamma, predictor=kernel_predictor, max_iters=max_iter)
    fair_clf.train(X, X_prime, y)

def multiple_comparision():
    ln_predictor = linear_model.LinearRegression()
    svm_predictor = svm.LinearSVR()
    tree_predictor = tree.DecisionTreeRegressor()
    kernel_predictor = KernelRidge(alpha=1.0, gamma=1.0, kernel='rbf')
    C = 15
    printflag = False
    predictor_dict = {'Linear': ln_predictor,
                      'SVR': svm_predictor,
                      'DT': tree_predictor,
                      'RBF Kernel': kernel_predictor}
    gamma_list = [0.01]
    max_iter_list = [5]
    results_dict = {}
    # For each model, train a Pareto curve
    for max_iter in max_iter_list:
        for curr_predictor in predictor_dict.keys():
            print('Curr Predictor: ')
            print(curr_predictor)
            predictor = predictor_dict[curr_predictor]
            fair_clf = Model(C=C, printflag=printflag, gamma=1, predictor=predictor, max_iters=max_iter)
            print(fair_clf.predictor)
            all_errors, all_fp = fair_clf.pareto(X, X_prime, y, gamma_list)
            results_dict[curr_predictor] = {'Errors' : all_errors, 'FP_disp': all_fp}

    print(results_dict)
    #pickle.dump(results_dict, open('results_max' + str(max_iter_list) + '_gammas' + str(gamma_list) + '.pkl', 'wb'))


def multiple_pareto():
    gamma_list = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]

    train_size = X.shape[0]
    X_train = X.iloc[:train_size]
    X_prime_train = X_prime.iloc[:train_size]
    y_train = y.iloc[:train_size]

    ln_predictor = linear_model.LinearRegression()
    svm_predictor = svm.LinearSVR()
    tree_predictor = tree.DecisionTreeRegressor(max_depth=3)
    kernel_predictor = KernelRidge(alpha=1.0, gamma=1.0, kernel='rbf')
    predictor_dict = {'Linear': {'predictor': ln_predictor, 'iters': 100},
                      'SVR': {'predictor': svm_predictor, 'iters': 10},
                      'DT': {'predictor': tree_predictor, 'iters': 100}}

    results_dict = {}

    for pred in predictor_dict:
        print('Curr Predictor: {}'.format(pred))
        predictor = predictor_dict[pred]['predictor']
        max_iters = predictor_dict[pred]['iters']
        fair_clf = Model(C=100, printflag=True, gamma=1, predictor=predictor, max_iters=max_iters)
        fair_clf.set_options(max_iters=max_iters)
        errors, fp_violations, fn_violations = fair_clf.pareto(X_train, X_prime_train, y_train, gamma_list)
        results_dict[pred] = {'errors': errors, 'fp_violations': fp_violations, 'fn_violations': fn_violations}
        plt.plot(errors, fp_violations, label=pred)

    pickle.dump(results_dict, open('results_dict_' + str(gamma_list) + '_gammas' + str(gamma_list) + '.pkl', 'wb'))

    
    plt.xlabel('Error')
    plt.ylabel('Unfairness')
    plt.legend()
    plt.title('Error vs. Unfairness\n(Communities & Crime Dataset)')
    plt.show()

multiple_pareto()