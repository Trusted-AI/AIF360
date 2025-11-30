from aif360.algorithms import Transformer
import numpy as np
import cvxpy as cp
from picos import Problem, RealVariable, trace
import warnings
import os
from ncpol2sdpa import SdpRelaxation, generate_operators, flatten

warnings.simplefilter(action='ignore', category=FutureWarning)

class _BaseLDS(Transformer):
    """
    Base class for Linear Dynamical System transformers.

    These algorithms have been adapted from [1]_.

    References:
        .. [1] Quan Zhou, Jakub Marecek, and Robert N. Shorten. “Fairness in Forecasting of Observations of Linear
            Dynamical Systems”. en. In: Journal of Artificial Intelligence Research 76 (Apr. 2023). arXiv:2209.05274 [cs,
            eess, math, stat], pp. 1247–1280. ISSN: 1076-9757. DOI: 10.1613/jair.1.14050. URL: http://arxiv.org/a
            bs/2209.05274 
    """
    def __init__(self, S, X, Y_hat, solver_path=None):
        """
        Base class for Linear Dynamical System transformers.

        The SDPA solver can be downloaded from https://sdpa.sourceforge.net/.
        This will provide a faster solution than using CVXPY, but requires the user to have the SDPA executable.

        Args:
            S (str): The name of the sensitive attribute.
            X (list): The names of the features.
            Y_hat (str): The name of the predicted label.
            solver_path (str): The path to the SDPA executable. If None, CVXPY will be used.
        """
        super().__init__()
        self.S = S
        self.X = X
        self.Y_hat = Y_hat
        self.solver_path = solver_path
        self.fitted = False

    def fit(self, dataset):
        """
        Fit the Linear Dynamical System transformer to the dataset.

        Args:
            dataset (BinaryLabelDataset): The dataset to fit the transformer to.
        Returns:
            LinearDynamicalSystem: Returns self.
        """
        D = dataset.convert_to_dataframe()[0]
        self._store_base_rate_info(D)
        D_privileged, D_unprivileged = self._split_data(D)
        constraints, obj_D, x_vars, e, z = self._create_optimisation(D_privileged, D_unprivileged)
        self.solved_x_vars, self.solved_e, self.solved_z = self._solve_optimisation(constraints, obj_D, x_vars, e, z)
        self.fitted = True
        return self

    def transform(self, dataset):
        """
        Transform the dataset using the fitted Linear Dynamical System transformer.

        Args:
            dataset (BinaryLabelDataset): The dataset to transform.
        Returns:
            BinaryLabelDataset: The transformed dataset.
        """
        if self.fitted:
            D = dataset.convert_to_dataframe()[0]
            D_reweighted = self._reweigh_data(D)
            D_normalised = self._normalise(D_reweighted)
            base_rate = self._calculate_base_rate(D)
            D_classified = self._apply_threshold(D_normalised, base_rate)
            dataset.df = D_classified
            return dataset.df
        else:
            raise Exception("Model has not been fitted yet")

    def fit_transform(self, dataset_train, dataset_test):
        """
        Fit the Linear Dynamical System transformer to the training dataset and transform the test dataset.

        Args:
            dataset_train (BinaryLabelDataset): The dataset to fit the transformer to.
            dataset_test (BinaryLabelDataset): The dataset to transform.
        Returns:
            BinaryLabelDataset: The transformed dataset.
        """
        self.fit(dataset_train)
        return self.transform(dataset_test)


    def _split_data(self, dataset):
        # Split the dataset into privileged and unprivileged groups based on the S attribute
        D_privileged = dataset[dataset[self.S] == 1]
        D_unprivileged = dataset[dataset[self.S] == 0]
        return D_privileged, D_unprivileged

    def _create_optimisation(self, D_privileged, D_unprivileged):
        x_vars, e, z = self._create_decision_variables()
        constraints, obj_D = self._get_constraints(D_privileged, D_unprivileged, x_vars, e, z)
        return constraints, obj_D, x_vars, e, z

    def _solve_optimisation(self, constraints, obj_D, x_vars, e, z):
        if self.solver_path != None:
            return self._solve_optimisation_sdpa(constraints, obj_D, x_vars, e, z)
        else:
            return self._solve_optimisation_cvx(constraints, obj_D, x_vars, e, z)

    def _solve_optimisation_sdpa(self, constraints, obj_D, x_vars, e, z):
        var_list = []
        for x in x_vars:
            var_list.extend(x)
        var_list.extend(e)
        var_list.extend(z)
        sdp_D = SdpRelaxation(variables = flatten(var_list), verbose = 0)
        sdp_D.get_relaxation(1, objective=obj_D, inequalities=constraints)
        sdp_D.solve(solver='sdpa', solverparameters={"executable":"sdpa_gmp","executable":self.solver_path})
        
        solved_e = [sdp_D[e[0]], sdp_D[e[1]]]
        solved_z = [sdp_D[z[0]], sdp_D[z[1]]]
        solved_x_vars = flatten([[sdp_D[i[0]], sdp_D[i[1]]] for i in x_vars])
        return solved_x_vars, solved_e, solved_z

    def _solve_optimisation_cvx(self, constraints, obj_D, x_vars, e, z):
        prob = Problem()
        prob.add_list_of_constraints(constraints)
        prob.set_objective('min', obj_D)

        try:
            prob.solve(solver='cvxopt', verbose=0)
        except Exception as exception:
            raise Exception(f"Optimisation failed to find a feasible solution. Error: {str(exception)}")

        solved_x_vars = flatten([[x[0].value, x[1].value] for x in x_vars])
        solved_e = [e[0].value, e[1].value]
        #iterate through z to get the values, as it is not a constant length
        solved_z = [z[i].value for i in range(len(z))]
        return solved_x_vars, solved_e, solved_z

    def _reweigh_data(self, D):
        D_reweighted = D.copy()
        
        for index, row in D_reweighted.iterrows():
            if row[self.S] == 0:
                score = sum(self.solved_x_vars[i] * row[x_name] for i, x_name in zip(range(0, len(self.solved_x_vars), 2), self.X)) + self.solved_e[0]
            else:
                score = sum(self.solved_x_vars[i] * row[x_name] for i, x_name in zip(range(1, len(self.solved_x_vars), 2), self.X)) + self.solved_e[1]
            D_reweighted.at[index, self.Y_hat] = score
        return D_reweighted
            
    def _normalise(self, D):
        D_normalised = D.copy()
        Y_hat_col = D_normalised[self.Y_hat]
        Y_hat_min = np.min(Y_hat_col)
        Y_hat_max = np.max(Y_hat_col)
        Y_hat_normalised = np.array([round(float(x - Y_hat_min) / (Y_hat_max - Y_hat_min), 1) for x in Y_hat_col])
        Y_hat_normalised[Y_hat_normalised > 1] = 1
        Y_hat_normalised[Y_hat_normalised < 0] = 0
        D_normalised[self.Y_hat] = Y_hat_normalised
        return D_normalised

    def _store_base_rate_info(self, dataset):
        D_privileged, D_unprivileged = self._split_data(dataset)
        self.priv_count = D_privileged.shape[0]
        self.priv_pos_count = D_privileged[D_privileged[self.Y_hat] == 1].shape[0]
        self.unpriv_count = D_unprivileged.shape[0]
        self.unpriv_pos_count = D_unprivileged[D_unprivileged[self.Y_hat] == 1].shape[0]

    def _calculate_base_rate(self, dataset):
        D_privileged, D_unprivileged = self._split_data(dataset)
        priv_count = D_privileged.shape[0] + self.priv_count
        priv_pos_count = D_privileged[D_privileged[self.Y_hat] == 1].shape[0] + self.priv_pos_count
        unpriv_count = D_unprivileged.shape[0] + self.unpriv_count
        unpriv_pos_count = D_unprivileged[D_unprivileged[self.Y_hat] == 1].shape[0] + self.unpriv_pos_count
        base_rate_privileged = 1- priv_pos_count / priv_count
        base_rate_unprivileged = 1- unpriv_pos_count / unpriv_count
        return base_rate_privileged, base_rate_unprivileged

    def _apply_threshold(self, dataset, base_rate):
        # Apply threshold based on base rate
        D_classified = dataset.copy()
        th_privileged = np.percentile(dataset[dataset[self.S] == 1][self.Y_hat], [base_rate[0]*100])[0]
        th_unprivileged = np.percentile(dataset[dataset[self.S] == 0][self.Y_hat], [base_rate[1]*100])[0]
        D_classified.loc[dataset[self.S] == 1, self.Y_hat] = (dataset[dataset[self.S] == 1][self.Y_hat] >= th_privileged).astype(int)
        D_classified.loc[dataset[self.S] == 0, self.Y_hat] = (dataset[dataset[self.S] == 0][self.Y_hat] >= th_unprivileged).astype(int)
        return D_classified
    
class SubgroupFairOptimiser(_BaseLDS):
    """
    The subgroup fair optimiser uses a min-max strategy minimise the loss 
    between a protected and unprotected subgroup, over the entire time period.
    """
    def _create_decision_variables(self):
        if self.solver_path != None:
            return self._create_decision_variables_sdpa()
        else:
            return self._create_decision_variables_cvx()

    def _create_decision_variables_sdpa(self):
        #for x in self.X, there is a 0 and 1 decision variable
        x_vars = [generate_operators(f"x{i}", n_vars=2, hermitian=True, commutative=False) for i in range(len(self.X))]
        e = generate_operators("e", n_vars=2, hermitian=True, commutative=False)
        z = generate_operators("z", n_vars=3, hermitian=True, commutative=True)
        return x_vars, e, z

    def _create_decision_variables_cvx(self):
        #for x in self.X, there is a 0 and 1 decision variable
        x_vars = [RealVariable(f"x{i}", (2,)) for i in range(len(self.X))]
        e = RealVariable("e", (2,))
        z = RealVariable("z", (3,))
        return x_vars, e, z

    def _get_constraints(self, D_privileged, D_unprivileged, x_vars, e, z):
        if self.solver_path != None:
            return self._get_constraints_sdpa(D_privileged, D_unprivileged, x_vars, e, z)
        else:
            return self._get_constraints_cvx(D_privileged, D_unprivileged, x_vars, e, z)

    def _get_constraints_sdpa(self, D_privileged, D_unprivileged, x_vars, e, z):
        ine1 = [z[0] + row[self.Y_hat] - sum(x[0] * row[x_name] for x, x_name in zip(x_vars, self.X)) + e[0] for _, row in D_unprivileged.iterrows()]
        ine2 = [z[0] - row[self.Y_hat] + sum(x[0] * row[x_name] for x, x_name in zip(x_vars, self.X)) + e[0] for _, row in D_unprivileged.iterrows()]
        ine3 = [z[0] + row[self.Y_hat] - sum(x[1] * row[x_name] for x, x_name in zip(x_vars, self.X)) + e[1] for _, row in D_privileged.iterrows()]
        ine4 = [z[0] - row[self.Y_hat] + sum(x[1] * row[x_name] for x, x_name in zip(x_vars, self.X)) + e[1] for _, row in D_privileged.iterrows()]
        max1 = [z[1] - sum((row[self.Y_hat] - sum(x[0] * row[x_name] for x, x_name in zip(x_vars, self.X)) + e[0])**2 for _, row in D_unprivileged.iterrows()) / len(D_unprivileged)]
        max2 = [z[2] - sum((row[self.Y_hat] - sum(x[1] * row[x_name] for x, x_name in zip(x_vars, self.X)) + e[1])**2 for _, row in D_privileged.iterrows()) / len(D_privileged)]

        obj_D = z[0] + z[1] + z[2] + 0.5 * (z[2] - z[1])
        constraints = ine1 + ine2 + ine3 + ine4 + max1 + max2
        return constraints, obj_D

    def _get_constraints_cvx(self, D_privileged, D_unprivileged, x_vars, e, z):
        constraints = []

        ine1 = [z[0] + row[self.Y_hat] - sum(x[0] * row[x_name] for x, x_name in zip(x_vars, self.X)) + e[0] for _, row in D_unprivileged.iterrows()]
        ine2 = [z[0] - row[self.Y_hat] + sum(x[0] * row[x_name] for x, x_name in zip(x_vars, self.X)) + e[0] for _, row in D_unprivileged.iterrows()]
        ine3 = [z[0] + row[self.Y_hat] - sum(x[1] * row[x_name] for x, x_name in zip(x_vars, self.X)) + e[1] for _, row in D_privileged.iterrows()]
        ine4 = [z[0] - row[self.Y_hat] + sum(x[1] * row[x_name] for x, x_name in zip(x_vars, self.X)) + e[1] for _, row in D_privileged.iterrows()]

        constraints.extend([ine >= 0 for ine in ine1 + ine2 + ine3 + ine4])

        max1 = z[1] - sum((row[self.Y_hat] - sum(x[0] * row[x_name] for x, x_name in zip(x_vars, self.X)) + e[0])**2 for _, row in D_unprivileged.iterrows()) / len(D_unprivileged)
        max2 = z[2] - sum((row[self.Y_hat] - sum(x[1] * row[x_name] for x, x_name in zip(x_vars, self.X)) + e[1])**2 for _, row in D_privileged.iterrows()) / len(D_privileged)

        constraints.append(max1 >= 0)
        constraints.append(max2 >= 0)

        obj_D = z[0] + z[1] + z[2] + 0.5 * (z[2] - z[1]) + 0.05*(e[0]**2 + e[1]**2)

        return constraints, obj_D

class InstantaneousFairOptimiser(_BaseLDS):
    """
    The instantaneous fair optimiser uses a min-max strategy minimise the loss 
    between a protected and unprotected subgroup, at each point in time.
    """
    def _create_decision_variables(self):
        if self.solver_path != None:
            return self._create_decision_variables_sdpa()
        else:
            return self._create_decision_variables_cvx()

    def _create_decision_variables_sdpa(self):
        #for x in self.X, there is an 0 and 1 decision variable
        x_vars = [generate_operators(f"x{i}", n_vars=2, hermitian=True, commutative=False) for i in range(len(self.X))]
        e = generate_operators("e", n_vars=2, hermitian=True, commutative=False)
        z = generate_operators("z", n_vars=2, hermitian=True, commutative=True)
        return x_vars, e, z

    def _create_decision_variables_cvx(self):
        #for x in self.X, there is an 0 and 1 decision variable
        x_vars = [RealVariable(f"x{i}", (2,)) for i in range(len(self.X))]
        e = RealVariable("e", (2,))
        z = RealVariable("z", (2,))
        return x_vars, e, z
        
    def _get_constraints(self, D_privileged, D_unprivileged, x_vars, e, z):
        if self.solver_path != None:
            return self._get_constraints_sdpa(D_privileged, D_unprivileged, x_vars, e, z)
        else:
            return self._get_constraints_cvx(D_privileged, D_unprivileged, x_vars, e, z)

    def _get_constraints_sdpa(self, D_privileged, D_unprivileged, x_vars, e, z):
        ine1 = [(z[0] + row[self.Y_hat] - sum(x[0] * row[x_name] for x, x_name in zip(x_vars, self.X)) + e[0])/len(D_unprivileged) for _, row in D_unprivileged.iterrows()]
        ine2 = [(z[0] - row[self.Y_hat] + sum(x[0] * row[x_name] for x, x_name in zip(x_vars, self.X)) + e[0])/len(D_unprivileged) for _, row in D_unprivileged.iterrows()]
        ine3 = [(z[0] + row[self.Y_hat] - sum(x[1] * row[x_name] for x, x_name in zip(x_vars, self.X)) + e[1])/len(D_privileged) for _, row in D_privileged.iterrows()]
        ine4 = [(z[0] - row[self.Y_hat] + sum(x[1] * row[x_name] for x, x_name in zip(x_vars, self.X)) + e[1])/len(D_privileged) for _, row in D_privileged.iterrows()]
        max1 = [(z[1] + row[self.Y_hat] - sum(x[0] * row[x_name] for x, x_name in zip(x_vars, self.X)) + e[0])/len(D_unprivileged) for _, row in D_unprivileged.iterrows()]
        max2 = [(z[1] - row[self.Y_hat] - sum(x[1] * row[x_name] for x, x_name in zip(x_vars, self.X)) + e[1])/len(D_privileged) for _, row in D_privileged.iterrows()]

        obj_D = z[0] + z[1]
        constraints = ine1 + ine2 + ine3 + ine4 + max1 + max2
        
        return constraints, obj_D

    def _get_constraints_cvx(self, D_privileged, D_unprivileged, x_vars, e, z):
        constraints = []

        ine1 = [(z[0] + row[self.Y_hat] - sum(x[0] * row[x_name] for x, x_name in zip(x_vars, self.X)) + e[0]) / len(D_unprivileged) for _, row in D_unprivileged.iterrows()]
        ine2 = [(z[0] - row[self.Y_hat] + sum(x[0] * row[x_name] for x, x_name in zip(x_vars, self.X)) + e[0]) / len(D_unprivileged) for _, row in D_unprivileged.iterrows()]
        ine3 = [(z[0] + row[self.Y_hat] - sum(x[1] * row[x_name] for x, x_name in zip(x_vars, self.X)) + e[1]) / len(D_privileged) for _, row in D_privileged.iterrows()]
        ine4 = [(z[0] - row[self.Y_hat] + sum(x[1] * row[x_name] for x, x_name in zip(x_vars, self.X)) + e[1]) / len(D_privileged) for _, row in D_privileged.iterrows()]

        constraints.extend([ine >= 0 for ine in ine1 + ine2 + ine3 + ine4])

        max1 = [(z[1] + row[self.Y_hat] - sum(x[0] * row[x_name] for x, x_name in zip(x_vars, self.X)) + e[0]) / len(D_unprivileged) for _, row in D_unprivileged.iterrows()]
        max2 = [(z[1] - row[self.Y_hat] - sum(x[1] * row[x_name] for x, x_name in zip(x_vars, self.X)) + e[1]) / len(D_privileged) for _, row in D_privileged.iterrows()]

        constraints.extend([max1_ine >= 0 for max1_ine in max1])
        constraints.extend([max2_ine >= 0 for max2_ine in max2])

        obj_D = z[0] + z[1] + 0.01*(e[0]**2 + e[1]**2)

        return constraints, obj_D