import pickle
import gerryfair
import matplotlib.pyplot as plt

def multiple_pareto():
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
    y_train_inv = [abs(1-y) for y in y_train]

    fp_auditor = gerryfair.model.Auditor(X_prime_train, y_train, 'FP')
    fn_auditor = gerryfair.model.Auditor(X_prime_train, y_train_inv, 'FP')

    comm_fp_violations = []
    comm_fn_violations = []

    for g in gamma_list:
        fair_model.set_options(gamma=g)
        fair_model.train(X_train, X_prime_train, y_train)
        predictions = fair_model.predict(X_train)
        predictions_inv = [abs(1-p) for p in predictions]
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

#multiple_pareto()
#fp_vs_fn()



