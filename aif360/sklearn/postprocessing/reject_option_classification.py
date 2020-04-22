import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from aif360.sklearn.metrics import statistical_parity_difference
from aif360.sklearn.metrics import average_odds_error
from aif360.sklearn.metrics import equal_opportunity_difference
from aif360.sklearn.metrics import disparate_impact_ratio
from aif360.sklearn.metrics import make_difference_scorer, make_ratio_scorer
from aif360.sklearn.utils import check_inputs, check_groups


class RejectOptionClassifier(BaseEstimator, ClassifierMixin):
    """Reject option based classification (ROC) post-processor.

    Reject option classification is a post-processing technique that gives
    favorable outcomes to unprivileged groups and unfavorable outcomes to
    privileged groups in a confidence band around the decision boundary with the
    highest uncertainty [#kamiran12]_.

    Note:
        A :class:`~sklearn.pipeline.Pipeline` expects a single estimation step
        but this class requires an estimator's predictions as input. See
        :class:`PostProcessingMeta` for a workaround.

    See also:
        :class:`PostProcessingMeta`, :class:`RejectOptionClassifierCV`

    References:
        .. [#kamiran12] `F. Kamiran, A. Karim, and X. Zhang, "Decision Theory
           for Discrimination-Aware Classification," IEEE International
           Conference on Data Mining, 2012.
           <https://ieeexplore.ieee.org/abstract/document/6413831>`_

    Attributes:
        prot_attr_ (str or list(str)): Protected attribute(s) used for post-
            processing.
        groups_ (array, shape (2,)): A list of group labels known to the
            classifier. Note: this algorithm require a binary division of the
            data.
        classes_ (array, shape (num_classes,)): A list of class labels known to
            the classifier. Note: this algorithm treats all non-positive
            outcomes as negative (binary classification only).
        pos_label_ (scalar): The label of the positive class.
        priv_group_ (scalar): The label of the privileged group.

    Examples:
        RejectOptionClassifier can be easily paired with GridSearchCV to
        find the best threshold and margin with respect to a fairness measure:

        >>> from sklearn.model_selection import GridSearchCV
        >>> roc = RejectOptionClassifier()
        >>> param = [{'threshold': [t],
                      'margin': np.linspace(0, min(t, 1-t), 50, endpoint=False)}
        ...          for t in np.arange(0.01, 1., 0.01)]
        >>> stat_par = make_difference_scorer(statistical_parity_difference)
        >>> scoring = {'bal_acc': 'balanced_accuracy', 'stat_par': stat_par}
        >>> def refit(cv_res):
        ...     return np.ma.array(cv_res['mean_test_bal_acc'],
        ...             mask=cv_res['mean_test_stat_par'] > 0.05).argmax()
        ...
        >>> grid = GridSearchCV(roc, param, scoring=scoring, refit=refit)

    """
    def __init__(self, prot_attr=None, threshold=0.5, margin=0.1):
        """
        Args:
            prot_attr (single label or list-like, optional): Protected
                attribute(s) to use in the post-processing. If more than one
                attribute, all combinations of values (intersections) are
                considered. Default is ``None`` meaning all protected attributes
                from the dataset are used. Note: This algorithm requires there
                be exactly 2 groups (privileged and unprivileged).
            threshold ():
            margin ():
            metric ('statistical_parity', 'average_odds', 'equal_opportunity',
                or callable):
        """
        self.prot_attr = prot_attr
        self.threshold = threshold
        self.margin = margin

    def fit(self, X, y, labels=None, pos_label=1, priv_group=1,
            sample_weight=None):
        """This is essentially a no-op; it simply validates the inputs and
        stores them for predict.

        Args:
            X (array-like): Probability estimates of the targets as
                returned by a ``predict_proba()`` call or equivalent.
            y (pandas.Series): Ground-truth (correct) target values.
            labels (list, optional): The ordered set of labels values. Must
                match the order of columns in X if provided. By default,
                all labels in y are used in sorted order.
            pos_label (scalar, optional): The label of the positive class.
            priv_group (scalar, optional): The label of the privileged group.
            sample_weight (array-like, optional): Sample weights.

        Returns:
            self
        """
        X, y, sample_weight = check_inputs(X, y, sample_weight)
        groups, self.prot_attr_ = check_groups(y, self.prot_attr,
                                               ensure_binary=True)
        self.classes_ = labels if labels is not None else np.unique(y)
        self.groups_ = np.unique(groups)
        self.pos_label_ = pos_label
        self.priv_group_ = priv_group

        if len(self.classes_) > 2:
            raise ValueError('Only binary classification is supported.')

        if pos_label not in self.classes_:
            raise ValueError('pos_label={} is not in the set of labels. The '
                    'valid values are:\n{}'.format(pos_label, self.classes_))

        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError('threshold must be between 0.0 and 1.0, '
                             'threshold={}'.format(self.threshold))

        if not 0.0 <= self.margin <= min(self.threshold, 1 - self.threshold):
            raise ValueError('margin must be between 0.0 and {}, margin={}'
                             ''.format(min(self.threshold, 1 - self.threshold),
                                       self.margin))

        return self

    def predict(self, X):
        """Predict class labels for the given scores.

        Args:
            X (pandas.DataFrame): Probability estimates of the targets as
                returned by a ``predict_proba()`` call or equivalent. Note: must
                include protected attributes in the index.

        Returns:
            numpy.ndarray: Predicted class label per sample.
        """
        check_is_fitted(self, 'pos_label_')

        groups, _ = check_groups(X, self.prot_attr_)
        if not set(np.unique(groups)) <= set(self.groups_):
            raise ValueError('The protected groups from X:\n{}\ndo not '
                             'match those from the training set:\n{}'.format(
                                     np.unique(groups), self.groups_))

        pos_idx = np.nonzero(self.classes_ == self.pos_label_)[0][0]
        X = X.iloc[:, pos_idx]

        yt = (X > self.threshold).astype(int)
        y_pred = self.classes_[yt if pos_idx == 1 else 1 - yt]

        # indices of critical region around the classification boundary
        crit_region = (abs(X - self.threshold) < self.margin)

        # replace labels in critical region
        neg_label = self.classes_[(pos_idx + 1) % 2]
        y_pred[crit_region & (groups == self.priv_group_)] = neg_label
        y_pred[crit_region & (groups != self.priv_group_)] = self.pos_label_

        return y_pred


class RejectOptionClassifierCV(GridSearchCV):
    """Wrapper for running a grid search over threshold, margin combinations for
    a RejectOptionClassifier.

    See also:
        :class:`RejectOptionClassifier`

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import pandas as pd
        >>> from sklearn.linear_model import LogisticRegression
        >>> from aif360.sklearn.datasets import fetch_german
        >>> from aif360.sklearn.postprocessing import RejectOptionClassifierCV
        >>> X, y = fetch_german(numeric_only=True)
        >>> lr = LogisticRegression(solver='lbfgs').fit(X, y)
        >>> roc = RejectOptionClassifierCV('disparate_impact', prot_attr='sex')
        >>> roc.fit(pd.DataFrame(lr.predict_proba(X), index=X.index), y)
        >>> res = pd.DataFrame(roc.cv_results_)
        >>> ax = res.plot.scatter('mean_test_disparate_impact', 'mean_test_bal_acc')
        >>> res.loc[[roc.best_index_]].plot.scatter('mean_test_disparate_impact', 'mean_test_bal_acc', color='r', ax=ax)
        >>> plt.show()

        We can also
    """
    def __init__(self, scorer, mask_func=None, step=0.05, prot_attr=None, **kwargs):
        """
        Args:
            scorer ('statistical_parity', 'average_odds', 'equal_opportunity',
                'disparate_impact', or callable):
            mask_func (callable, optional): A
            step (float): Step size for grid search.
            prot_attr (single label or list-like, optional): Protected
                attribute(s) to use in the post-processing. If more than one
                attribute, all combinations of values (intersections) are
                considered. Default is ``None`` meaning all protected attributes
                from the dataset are used. Note: This algorithm requires there
                be exactly 2 groups (privileged and unprivileged).
        """
        self.scorer = scorer
        self.mask_func = mask_func
        self.step = step
        self.prot_attr = prot_attr
        super().__init__(RejectOptionClassifier(), {}, **kwargs)

    def fit(self, X, y, **fit_params):
        self.param_grid = [
                {'prot_attr': [self.prot_attr], 'threshold': [t],
                 'margin': np.arange(self.step, min(t, 1-t), self.step)}
                for t in np.arange(self.step, 1, self.step)]

        self.scorer_name_ = self.scorer
        if self.scorer == 'statistical_parity':
            self.scorer_ = make_difference_scorer(statistical_parity_difference,
                    prot_attr=self.prot_attr)
        elif self.scorer == 'average_odds':
            self.scorer_ = make_difference_scorer(average_odds_error,
                    prot_attr=self.prot_attr)
        elif self.scorer == 'equal_opportunity':
            self.scorer_ = make_difference_scorer(equal_opportunity_difference,
                    prot_attr=self.prot_attr)
        elif self.scorer == 'disparate_impact':
            self.scorer_ = make_ratio_scorer(disparate_impact_ratio,
                    prot_attr=self.prot_attr)
        elif not callable(self.scorer):
            raise ValueError("scorer must be one of: 'statistical_parity', "
                "'average_odds', 'equal_opportunity', 'disparate_impact' or a "
                "callable function. Got:\n{}".format(self.scorer))
        else:
            self.scorer_name_ = 'fairness_metric'
            self.scorer_ = self.scorer_

        self.scoring = {'bal_acc': 'balanced_accuracy',
                        self.scorer_name_: self.scorer_}

        if self.refit is True:
            if self.mask_func is None:
                if self.scorer_name_ == 'fairness_metric':
                    self.refit = False
                elif self.scorer_name_ == 'disparate_impact':
                    self.mask_func_ = lambda x: x < 0.8
                else:
                    self.mask_func_ = lambda x: x < -0.1
            else:
                self.mask_func_ = self.mask_func
        if self.refit is True:
            self.refit = lambda cvr: np.ma.array(
                    cvr['mean_test_bal_acc'],
                    mask=self.mask_func_(cvr['mean_test_'+self.scorer_name_])
            ).argmax()

        super().fit(X, y, **fit_params)
