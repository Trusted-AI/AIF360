import gerryfair

dataset = "./dataset/communities.csv"
attributes = "./dataset/communities_protected.csv"
centered = True
X, X_prime, y = gerryfair.clean.clean_dataset(dataset, attributes, centered)
C = 10
printflag = True
gamma = .01
fair_model = gerryfair.model.Model(C=C, printflag=printflag, gamma=gamma, fairness_def='FP')
max_iters = 5
fair_model.set_options(max_iters=max_iters)

# Train Set
train_size = 300

X_train = X.iloc[:train_size]
X_prime_train = X_prime.iloc[:train_size]
y_train = y.iloc[:train_size]

# Test Set
X_test = X.iloc[train_size:].reset_index(drop=True)
X_prime_test = X_prime.iloc[train_size:].reset_index(drop=True)
y_test = y.iloc[train_size:].reset_index(drop=True)

# Train the model
[errors, fp_difference] = fair_model.train(X_train, X_prime_train, y_train)

# Generate predictions
predictions = fair_model.predict(X_train)

# Audit predictions
auditor = gerryfair.model.Auditor(X_prime_train, y_train, 'FP')
[group, fairness_violation] = auditor.audit(predictions)
print(fairness_violation)