from aif360.algorithms.inprocessing.gerryfair_classifier import Model
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
from aif360.algorithms.inprocessing import gerryfair
from sklearn import svm
from sklearn import tree
from sklearn.kernel_ridge import KernelRidge
from sklearn import linear_model
import pickle
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    print("Matplotlib Error, comment out matplotlib.use('TkAgg')")
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


def multiple_classifiers_pareto(dataset, gamma_list=[0.002, 0.005, 0.01, 0.02, 0.05, 0.1], save_results=True):


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


def multiple_datasets_pareto():
    communities_dataset = "./dataset/communities.csv"
    communities_attributes = "./dataset/communities_protected.csv"
    lawschool_dataset = "./dataset/lawschool.csv"
    lawschool_attributes = "./dataset/lawschool_protected.csv"
    adult_dataset = "./dataset/adult.csv"
    adult_attributes = "./dataset/adult_protected.csv"
    student_dataset = "./dataset/student-mat.csv"
    student_attributes = "./dataset/student_protected.csv"

    C = 10
    printflag = True
    gamma = .01
    fair_model = gerryfair.model.Model(C=C, printflag=printflag, gamma=gamma, fairness_def='FP')
    gamma_list = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    centered = True

    # Train Set (Communities)
    X, X_prime, y = gerryfair.clean.clean_dataset(communities_dataset, communities_attributes, centered)

    # Train the model (size=1000, iters=200)
    train_size = 200
    max_iters = 1000
    X_train = X.iloc[:train_size]
    X_prime_train = X_prime.iloc[:train_size]
    y_train = y.iloc[:train_size]
    fair_model.set_options(max_iters=max_iters)
    communities_all_errors, communities_all_violations = fair_model.pareto(X_train, X_prime_train, y_train, gamma_list)

    # Train Set (Law School)
    X, X_prime, y = gerryfair.clean.clean_dataset(lawschool_dataset, lawschool_attributes, centered)

    # Train the model
    train_size = 200
    max_iters = 1000
    X_train = X.iloc[:train_size]
    X_prime_train = X_prime.iloc[:train_size]
    y_train = y.iloc[:train_size]
    fair_model.set_options(max_iters=max_iters)
    lawschool_all_errors, lawschool_all_violations = fair_model.pareto(X_train, X_prime_train, y_train, gamma_list)

    # Train Set (Adult Income)
    X, X_prime, y = gerryfair.clean.clean_dataset(adult_dataset, adult_attributes, centered)

    # Train the model
    train_size = 200
    max_iters = 1000
    X_train = X.iloc[:train_size]
    X_prime_train = X_prime.iloc[:train_size]
    y_train = y.iloc[:train_size]
    fair_model.set_options(max_iters=max_iters)
    adult_all_errors, adult_all_violations = fair_model.pareto(X_train, X_prime_train, y_train, gamma_list)

    # Train Set (Adult Income)
    X, X_prime, y = gerryfair.clean.clean_dataset(student_dataset, student_attributes, centered)

    # Train the model
    train_size = 200
    max_iters = 1000
    X_train = X.iloc[:train_size]
    X_prime_train = X_prime.iloc[:train_size]
    y_train = y.iloc[:train_size]
    fair_model.set_options(max_iters=max_iters)
    student_all_errors, student_all_violations = fair_model.pareto(X_train, X_prime_train, y_train, gamma_list)

    plt.plot(communities_all_errors, communities_all_violations, label='communities')
    plt.plot(lawschool_all_errors, lawschool_all_violations, label='law school')
    plt.plot(adult_all_errors, adult_all_violations, label='adult')
    plt.plot(student_all_errors, student_all_violations, label='student')
    plt.xlabel('error')
    plt.ylabel('unfairness')
    plt.legend()
    plt.title('error vs. unfairness (iterations = {})'.format(max_iters))
    plt.show()

    print("done")


def fp_vs_fn():
    communities_dataset = "./dataset/communities.csv"
    communities_attributes = "./dataset/communities_protected.csv"
    C = 10
    printflag = True
    gamma = .01
    max_iters = 50
    fair_model = gerryfair.model.Model(C=C, printflag=printflag, gamma=gamma, fairness_def='FP', max_iters=max_iters)
    gamma_list = [0.001, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.05]
    centered = True

    # Train Set (Communities)
    X, X_prime, y = gerryfair.clean.clean_dataset(communities_dataset, communities_attributes, centered)

    # Train the model (size=1000, iters=200)
    train_size = 1900
    X_train = X.iloc[:train_size]
    X_prime_train = X_prime.iloc[:train_size]
    y_train = y.iloc[:train_size]
    y_train_inv = [abs(1 - y) for y in y_train]

    fp_auditor = gerryfair.model.Auditor(X_prime_train, y_train, 'FP')
    fn_auditor = gerryfair.model.Auditor(X_prime_train, y_train_inv, 'FP')

    fp_violations = []
    fn_violations = []

    for g in gamma_list:
        fair_model.set_options(gamma=g)
        fair_model.train(X_train, X_prime_train, y_train)
        predictions = fair_model.predict(X_train)
        predictions_inv = [abs(1 - p) for p in predictions]
        _, fp_diff = fp_auditor.audit(predictions)
        _, fn_diff = fn_auditor.audit(predictions_inv)
        fp_violations.append(fp_diff)
        fn_violations.append(fn_diff)

    print((fp_violations, fn_violations))

    plt.plot(fp_violations, fn_violations, label='communities')
    plt.xlabel('False Positive Disparity')
    plt.ylabel('False Negative Disparity')
    plt.legend()
    plt.title('FP vs FN Unfairness')
    plt.show()






if __name__ == '__main__':

    dataset = load_preproc_data_adult()
    multiple_classifiers_pareto(dataset)
    multiple_datasets_pareto()
    fp_vs_fn()
