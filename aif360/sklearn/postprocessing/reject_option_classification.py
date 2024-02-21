import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_is_fitted

from aif360.sklearn.metrics import statistical_parity_difference
from aif360.sklearn.metrics import average_odds_error
from aif360.sklearn.metrics import equal_opportunity_difference
from aif360.sklearn.metrics import disparate_impact_ratio
from aif360.sklearn.metrics import make_scorer
from aif360.sklearn.utils import check_groups


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
                      'margin': np.arange(0.05, min(t, 1-t)+0.025, 0.05)}
        ...          for t in np.arange(0.05, 1., 0.05)]
        >>> stat_par = make_scorer(statistical_parity_difference)
        >>> scoring = {'bal_acc': 'balanced_accuracy', 'stat_par': stat_par}
        >>> def refit(cv_res):
        ...     return np.ma.array(cv_res['mean_test_bal_acc'],
        ...             mask=cv_res['mean_test_stat_par'] < -0.1).argmax()
        ...
        >>> grid = GridSearchCV(roc, param, scoring=scoring, refit=refit)

        Or, alternatively, this can be done in one step with
        RejectOptionClassifierCV:

        >>> grid = RejectOptionClassifierCV(scoring='statistical_parity')

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
            threshold (scalar): Classification threshold. Probability estimates
                greater than this value are considered positive. Must be between
                0 and 1.
            margin (scalar): Half width of the critical region. Estimates within
                the critical region are "rejected" and assigned according to
                their group. Must be between 0 and min(threshold, 1-threshold).
        """
        self.prot_attr = prot_attr
        self.threshold = threshold
        self.margin = margin

    def _more_tags(self):
        return {'requires_proba': True}

    def fit(self, X, y, labels=None, pos_label=1, priv_group=1,
            sample_weight=None):
        """This is essentially a no-op; it simply validates the inputs and
        stores them for predict.

        Args:
            X (array-like): Ignored.
            y (array-like): Ground-truth (correct) target values. Note: one of X
                or y must contain protected attribute information.
            labels (list, optional): The ordered set of labels values. Must
                match the order of columns in X if provided. By default,
                all labels in y are used in sorted order.
            pos_label (scalar, optional): The label of the positive class.
            priv_group (scalar, optional): The label of the privileged group.
            sample_weight (array-like, optional): Ignored.

        Returns:
            self
        """
        try:
            groups, self.prot_attr_ = check_groups(X, self.prot_attr,
                                                   ensure_binary=True)
        except TypeError:
            groups, self.prot_attr_ = check_groups(y, self.prot_attr,
                                                   ensure_binary=True)
        self.classes_ = np.array(labels) if labels is not None else np.unique(y)
        self.groups_ = np.unique(groups)
        self.pos_label_ = pos_label
        self.priv_group_ = priv_group

        if len(self.classes_) != 2:
            raise ValueError('Only binary classification is supported.')

        if pos_label not in self.classes_:
            raise ValueError('pos_label={} is not in the set of labels. The '
                    'valid values are:\n{}'.format(pos_label, self.classes_))

        if priv_group not in self.groups_:
            raise ValueError('priv_group={} is not in the set of groups. The '
                    'valid values are:\n{}'.format(priv_group, self.groups_))

        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError('threshold must be between 0.0 and 1.0, '
                             'threshold={}'.format(self.threshold))

        if not 0.0 <= self.margin <= min(self.threshold, 1 - self.threshold):
            raise ValueError('margin must be between 0.0 and {}, margin={}'
                             ''.format(min(self.threshold, 1 - self.threshold),
                                       self.margin))

        return self

    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates for all classes are ordered by the label of
        classes.

        Args:
            X (pandas.DataFrame): Probability estimates of the targets as
                returned by a ``predict_proba()`` call or equivalent. Note: must
                include protected attributes in the index.
        Returns:
            numpy.ndarray: Returns the probability of the sample for each class
            in the model, where classes are ordered as they are in
            ``self.classes_``.
        """
        check_is_fitted(self, 'pos_label_')

        groups, _ = check_groups(X, self.prot_attr_)
        if len(self.classes_) != X.shape[1]:
            raise ValueError('X should contain one column per class. Got: {} '
                             'columns.'.format(X.shape[1]))

        pos_idx = np.nonzero(self.classes_ == self.pos_label_)[0][0]
        yt = X.iloc[:, pos_idx].to_numpy().copy()

        # indices of critical region around the classification boundary
        crit_above = (self.margin > yt-self.threshold) & (yt > self.threshold)
        crit_below = (-self.margin < yt-self.threshold) & (yt < self.threshold)

        # flip labels: priv + above -> below, unpriv + below -> above
        priv = (groups == self.priv_group_)
        flip = (priv & crit_above) | (~priv & crit_below)
        yt[flip] = 2*self.threshold - yt[flip]

        return np.c_[1 - yt, yt] if pos_idx == 1 else np.c_[yt, 1 - yt]

    def predict(self, X):
        """Predict class labels for the given scores.

        Args:
            X (pandas.DataFrame): Probability estimates of the targets as
                returned by a ``predict_proba()`` call or equivalent. Note: must
                include protected attributes in the index.

        Returns:
            numpy.ndarray: Predicted class label per sample.
        """
        scores = self.predict_proba(X)
        pos_idx = np.nonzero(self.classes_ == self.pos_label_)[0][0]
        y_pred = (scores[:, pos_idx] > self.threshold).astype(int)
        return self.classes_[y_pred if pos_idx == 1 else 1 - y_pred]

    def fit_predict(self, X, y=None, **fit_params):
        """Predict class labels for the given scores.

        In general, it is not necessary to fit and predict separately so this
        method may be used instead. For subsequent predicts, it may be easier
        to use the `predict` method, though.

        Args:
            X (pandas.DataFrame): Probability estimates of the targets as
                returned by a ``predict_proba()`` call or equivalent. Note: must
                include protected attributes in the index.
            y (array-like, optional): Ground-truth (correct) target values.
                Note: if not provided, `labels` must be provided in
                `**fit_params`. See `fit` for details.
            **fit_params: See `fit` for details.

        Returns:
            numpy.ndarray: Predicted class label per sample.
        """
        return self.fit(X, y, **fit_params).predict(X)


class RejectOptionClassifierCV(GridSearchCV):
    """Wrapper for running a grid search over threshold, margin combinations for
    a RejectOptionClassifier.

    Note:
        :class:`~sklearn.model_selection.GridSearchCV` does not currently
        support sample weights in scoring. This will work but throw a warning if
        `sample_weight` is provided.

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
        >>> roc = RejectOptionClassifierCV('sex', scoring='disparate_impact')
        >>> roc.fit(pd.DataFrame(lr.predict_proba(X), index=X.index), y)

        We can also achieve this more simply using a PostProcessingMeta
        estimator:

        >>> from aif360.sklearn.postprocessing import PostProcessingMeta
        >>> pp = PostProcessingMeta(lr, roc).fit(X, y)
    """
    def __init__(self, prot_attr=None, *, scoring, step=0.05, refit=True, **kwargs):
        """
        Args:
            prot_attr (single label or list-like, optional): Protected
                attribute(s) to use in the post-processing. If more than one
                attribute, all combinations of values (intersections) are
                considered. Default is ``None`` meaning all protected attributes
                from the dataset are used. Note: This algorithm requires there
                be exactly 2 groups (privileged and unprivileged).
            scoring ('statistical_parity', 'average_odds', 'equal_opportunity',
                'disparate_impact', or callable/dict): Fairness scorer to use to
                evaluate the predictions. If type is a `str`, constructs the
                corresponding scorer for that metric in addition to the default
                balanced accuracy. If type is callable (i.e., a scorer object),
                that will be used along with balanced accuracy. Finally, if an
                explicit dictionary is passed, this will be used as is.
            step (float): Step size for grid search. Will search every valid
                combination of threshold and margin that are multiples of this
                step size. See `param_grid` after fitting for the exact search
                space.
            refit (bool or callable, optional): Refit the estimator using the
                best parameters found. If `True` and not using a custom scoring
                function, this chooses the highest balanced accuracy given
                fairness score > -0.1 (or > 0.8 for disparate impact only).
                Alternatively, a custom refitting function may be passed. See
                :class:`~sklearn.model_selection.GridSearchCV` for details.
            **kwargs: See :class:`~sklearn.model_selection.GridSearchCV` for
                additional kwargs.
        """
        self.scoring = scoring
        self.refit = refit
        self.step = step
        self.prot_attr = prot_attr
        super().__init__(RejectOptionClassifier(), {}, scoring=scoring,
                         refit=refit, **kwargs)

    def _more_tags(self):
        return {'requires_proba': True}

    def fit(self, X, y, **fit_params):
        """Run fit with all sets of parameters.

        Args:
            X (pandas.DataFrame): Probability estimates of the targets as
                returned by a ``predict_proba()`` call or equivalent. Note: must
                include protected attributes in the index.
            y (pandas.Series): Ground-truth (correct) target values.
            **fit_params: Parameters passed to the ``fit()`` method.

        Returns:
            self
        """
        self.param_grid = []
        thresholds = np.arange(self.step, 1, self.step)
        # arange has numerical instabilities. this way guarantees margin <= threshold
        for i, t in enumerate(thresholds):
            n = min(i+1, len(thresholds)-i)
            margins = np.linspace(min(self.step, min(t, 1-t)), min(t, 1-t), n)
            self.param_grid.append({'prot_attr': [self.prot_attr],
                                    'threshold': [t], 'margin': margins})

        if fit_params.get('sample_weight', None) is not None:
            warnings.warn('sample_weight will be ignored when scoring.',
                          RuntimeWarning)

        if not isinstance(self.scoring, dict):
            # TODO: sample_weight scoring workaround
            self.scorer_name_ = self.scoring
            if self.scoring == 'statistical_parity':
                self.scorer_ = make_scorer(statistical_parity_difference,
                        prot_attr=self.prot_attr)
            elif self.scoring == 'average_odds':
                self.scorer_ = make_scorer(average_odds_error,
                        prot_attr=self.prot_attr)
            elif self.scoring == 'equal_opportunity':
                self.scorer_ = make_scorer(equal_opportunity_difference,
                        prot_attr=self.prot_attr)
            elif self.scoring == 'disparate_impact':
                self.scorer_ = make_scorer(disparate_impact_ratio, is_ratio=True,
                        prot_attr=self.prot_attr, zero_division=0)
            elif not callable(self.scoring):
                raise ValueError("scorer must be one of: 'statistical_parity', "
                    "'average_odds', 'equal_opportunity', 'disparate_impact' "
                    "or a callable function. Got:\n{}".format(self.scoring))
            else:
                self.scorer_name_ = 'fairness_metric'
                self.scorer_ = self.scoring

            self.scoring = {'bal_acc': 'balanced_accuracy',
                            self.scorer_name_: self.scorer_}

        if self.refit is True and self.scorer_name_ != 'fairness_metric':
            if self.scorer_name_ == 'disparate_impact':
                self.refit = lambda res: np.ma.array(res['mean_test_bal_acc'],
                    mask=res['mean_test_disparate_impact'] < 0.8).argmax()
            else:
                self.refit = lambda res: np.ma.array(res['mean_test_bal_acc'],
                    mask=res['mean_test_'+self.scorer_name_] < -0.1).argmax()

        class NoSplit:
            def split(self, X, y=None, groups=None):
                yield np.arange(len(X)), np.arange(len(X))

            def get_n_splits(self, X=None, y=None, groups=None):
                return 1

        self.cv = NoSplit()

        return super().fit(X, y, **fit_params)
