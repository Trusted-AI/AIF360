import numpy as np
import scipy.special
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
import tensorflow as tf
import random


from aif360.sklearn.utils import check_inputs, check_groups

class classifier_model(tf.Module):
    def __init__(self,feature,Hneuron1,output,dropout,seed1,seed2):
        super(classifier_model, self).__init__()
        self.feature = feature
        self.hN1 = Hneuron1
        self.output = output
        self.dropout = dropout
        self.seed1 = seed1
        self.seed2 = seed2
        self.W1 = tf.Variable(tf.random.uniform(shape=(self.feature,self.hN1), seed=self.seed1), name='W1')
        self.b1 = tf.Variable(tf.zeros(shape=self.hN1), name='b1')
        self.W2 = tf.Variable(tf.random.uniform(shape=(self.hN1,self.output), seed=self.seed2), name='W2')
        self.b2 = tf.Variable(tf.zeros(shape=self.output), name='b2')
        self.sigmoid = tf.nn.sigmoid

    def forward(self, x):
        x = tf.nn.relu(tf.matmul(x, self.W1) + self.b1)
        x = tf.nn.dropout(x, rate=self.dropout, seed=self.seed2)
        x_logits = tf.matmul(x, self.W2) + self.b2
        x_pred = tf.sigmoid(x_logits)
        return x_pred, x_logits

class adversary_model(tf.Module):
    def __init__(self, seed3, n_groups=1):
        super(adversary_model, self).__init__()
        self.seed3 = seed3
        self.c = tf.Variable(tf.constant(1.0), name = 'c')
        self.W1 = tf.Variable(tf.random.normal(shape=(3, n_groups),seed=self.seed3), name='w1')
        self.b1 = tf.Variable(tf.zeros(shape=n_groups), name = 'b1')
        self.sigmoid = tf.nn.sigmoid

    def forward(self,pred_logits, true_labels):
        s = self.sigmoid((1 + tf.abs(self.c)) * pred_logits)
        pred_protected_attribute_logits = tf.matmul(tf.concat([s, s * true_labels, s * (1.0 - true_labels)], axis=1), self.W1) + self.b1
        pred_protected_attribute_labels = tf.sigmoid(pred_protected_attribute_logits)
        return pred_protected_attribute_labels, pred_protected_attribute_logits


class AdversarialDebiasing(BaseEstimator, ClassifierMixin):
    """Debiasing with adversarial learning.

    Adversarial debiasing is an in-processing technique that learns a
    classifier to maximize prediction accuracy and simultaneously reduce an
    adversary's ability to determine the protected attribute from the
    predictions [#zhang18]_. This approach leads to a fair classifier as the
    predictions cannot carry any group discrimination information that the
    adversary can exploit.

    References:
        .. [#zhang18] `B. H. Zhang, B. Lemoine, and M. Mitchell, "Mitigating
           Unwanted Biases with Adversarial Learning," AAAI/ACM Conference on
           Artificial Intelligence, Ethics, and Society, 2018.
           <https://dl.acm.org/citation.cfm?id=3278779>`_

    Attributes:
        prot_attr_ (str or list(str)): Protected attribute(s) used for
            debiasing.
        groups_ (array, shape (n_groups,)): A list of group labels known to the
            classifier.
        classes_ (array, shape (n_classes,)): A list of class labels known to
            the classifier.
        sess_ (tensorflow.Session): The TensorFlow Session used for the
            computations. Note: this can be manually closed to free up resources
            with `self.sess_.close()`.
        classifier_logits_ (tensorflow.Tensor): Tensor containing output logits
            from the classifier.
        adversary_logits_ (tensorflow.Tensor): Tensor containing output logits
            from the adversary.
    """

    def __init__(self, prot_attr=None, scope_name='classifier',
                 adversary_loss_weight=0.1, num_epochs=50, batch_size=256,
                 classifier_num_hidden_units=200, debias=True, verbose=False,
                 random_state=None):
        r"""
        Args:
            prot_attr (single label or list-like, optional): Protected
                attribute(s) to use in the debiasing process. If more than one
                attribute, all combinations of values (intersections) are
                considered. Default is ``None`` meaning all protected attributes
                from the dataset are used.
            scope_name (str, optional): TensorFlow "variable_scope" name for the
                entire model (classifier and adversary).
            adversary_loss_weight (float or ``None``, optional): If ``None``,
                this will use the suggestion from the paper:
                :math:`\alpha = \sqrt(global_step)` with inverse time decay on
                the learning rate. Otherwise, it uses the provided coefficient
                with exponential learning rate decay.
            num_epochs (int, optional): Number of epochs for which to train.
            batch_size (int, optional): Size of mini-batch for training.
            classifier_num_hidden_units (int, optional): Number of hidden units
                in the classifier.
            debias (bool, optional): If ``False``, learn a classifier without an
                adversary.
            verbose (bool, optional): If ``True``, print losses every 200 steps.
            random_state (int or numpy.RandomState, optional): Seed of pseudo-
                random number generator for shuffling data and seeding weights.
        """

        self.prot_attr = prot_attr
        self.scope_name = scope_name
        self.adversary_loss_weight = adversary_loss_weight
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.classifier_num_hidden_units = classifier_num_hidden_units
        self.debias = debias
        self.verbose = verbose
        self.random_state = random_state
        self.features_dim = None
        self.features_ph = None
        self.protected_attributes_ph = None
        self.true_labels_ph = None
        self.pred_labels = None

    def fit(self, X, y):
        """Train the classifier and adversary (if ``debias == True``) with the
        given training data.

        Args:
            X (pandas.DataFrame): Training samples.
            y (array-like): Training labels.

        Returns:
            self
        """

        X, y, _ = check_inputs(X, y)
        rng = check_random_state(self.random_state)
        if self.random_state is not None:
            np.random.seed(self.random_state)
        ii32 = np.iinfo(np.int32)
        self.s1, self.s2, self.s3 = rng.randint(ii32.min, ii32.max, size=3)
        tf.random.set_seed(self.random_state)

        groups, self.prot_attr_ = check_groups(X, self.prot_attr)
        le = LabelEncoder()
        y = le.fit_transform(y)
        self.classes_ = le.classes_
        # BUG: LabelEncoder converts to ndarray which removes tuple formatting
        groups = groups.map(str)
        groups = le.fit_transform(groups)
        self.groups_ = le.classes_

        n_classes = len(self.classes_)
        n_groups = len(self.groups_)
        # use sigmoid for binary case
        if n_classes == 2:
            n_classes = 1
        if n_groups == 2:
            n_groups = 1
        if n_groups>2:
            # For intersection of more sensitive variable, i think this way
            # is not corrected! maybe should be prefered to build n different adversary model
            # instead of a unique handling both the sensitive classes
            OHE = OneHotEncoder()
            groups = OHE.fit_transform(groups.reshape(-1,1)).toarray()

        num_train_samples, n_features = X.shape

        starter_learning_rate = 0.001
        self.clf_model = classifier_model(feature=n_features, Hneuron1=self.classifier_num_hidden_units,
                                          output=1, dropout=0.2,
                                          seed1=self.s1, seed2=self.s2)
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(starter_learning_rate,
                                                                       decay_steps=1000, decay_rate=0.96,
                                                                       staircase=True)
        classifier_opt = tf.optimizers.Adam(learning_rate)
        classifier_vars = [var for var in self.clf_model.trainable_variables]

        # pretrain_both_models
        if self.debias:
            self.adv_model = adversary_model(seed3=self.s3, n_groups=n_groups)
            learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(starter_learning_rate,
                                                                           decay_steps=1000, decay_rate=0.96,
                                                                           staircase=True)
            adversary_opt = tf.optimizers.Adam(learning_rate)
            # adversary_vars = [var for var in self.adv_model.trainable_variables]
            for epoch in range(self.num_epochs//2):
                shuffled_ids = [i for i in range(num_train_samples) ]
                # BUG: np.random.choice not reproduce same shuffled id every epochs
                #shuffled_ids = np.random.choice(num_train_samples, num_train_samples, replace=False)
                for i in range(num_train_samples // self.batch_size):
                    batch_ids = shuffled_ids[self.batch_size * i: self.batch_size * (i + 1)]
                    batch_features = X.values[batch_ids].astype('float32')
                    batch_labels = np.reshape(y[batch_ids], [-1, 1]).astype('float32')
                    with tf.GradientTape() as tape:
                        pred_labels, pred_logits = self.clf_model.forward(batch_features)
                        loss_clf = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_labels, logits=pred_logits))
                    gradients = tape.gradient(loss_clf, self.clf_model.trainable_variables)
                    classifier_opt.apply_gradients(zip(gradients, self.clf_model.trainable_variables))
                    if i % 200 == 0:
                        print("(Pretraining Classifier) epoch %d; iter: %d; batch classifier loss: %f" % (
                            epoch, i, loss_clf))
            for epoch in range(self.num_epochs // 5):
                # BUG: np.random.choice not reproduce same shuffled id every epochs
                #shuffled_ids = np.random.choice(num_train_samples, num_train_samples, replace=False)
                for i in range(num_train_samples // self.batch_size):
                    batch_ids = shuffled_ids[self.batch_size * i: self.batch_size * (i + 1)]
                    batch_features = X.values[batch_ids].astype('float32')
                    batch_labels = np.reshape(y[batch_ids], [-1, 1]).astype('float32')
                    batch_protected_attributes = np.reshape(groups[batch_ids][:, np.newaxis],[-1,1]).astype('float32')
                    with tf.GradientTape() as tape:
                        pred_labels, pred_logits = self.clf_model.forward(batch_features)
                        pred_protected_attributes_labels, pred_protected_attributes_logits = self.adv_model.forward(
                            pred_logits, batch_labels)
                        loss_adv = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_protected_attributes.reshape(self.batch_size,n_groups),
                                                                    logits=pred_protected_attributes_logits))
                    gradients = tape.gradient(loss_adv, self.adv_model.trainable_variables)
                    adversary_opt.apply_gradients(zip(gradients, self.adv_model.trainable_variables))
                    if i % 200 == 0:
                        print("(Pretraining Adversarial Net) epoch %d; iter: %d; batch classifier loss: %f" % (
                            epoch, i, loss_clf))
            # Adversary Debiasing
            normalize = lambda x: x / (tf.norm(x) + np.finfo(np.float32).tiny)
            for epoch in range(self.num_epochs):
                shuffled_ids = [i for i in range(num_train_samples)]
                # self.clf_model.dropout=0
                # self.adversary_loss_weight=sqrt(epoch)
                # BUG: np.random.choice not reproduce same shuffled id every epochs
                #shuffled_ids = np.random.choice(num_train_samples, num_train_samples, replace=False)
                for i in range(num_train_samples // self.batch_size):
                    batch_ids = shuffled_ids[self.batch_size * i: self.batch_size * (i + 1)]
                    batch_features = X.values[batch_ids].astype('float32')
                    batch_labels = np.reshape(y[batch_ids], [-1, 1]).astype('float32')
                    batch_protected_attributes = np.reshape(groups[batch_ids][:, np.newaxis],[-1,1]).astype('float32')
                    with tf.GradientTape() as tape:
                        pred_labels, pred_logits = self.clf_model.forward(batch_features)
                        loss_clf = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_labels, logits=pred_logits))
                    classifier_grad = tape.gradient(loss_clf, classifier_vars)
                    classifier_grads = []

                    with tf.GradientTape() as tape1:
                        pred_labels, pred_logits = self.clf_model.forward(
                            batch_features)  # varaibles of CLF_model need to be watched from tape1 also
                        pred_protected_attributes_labels, pred_protected_attributes_logits = self.adv_model.forward(
                            pred_logits, batch_labels)
                        loss_adv = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_protected_attributes.reshape(self.batch_size,n_groups),
                                                                    logits=pred_protected_attributes_logits))
                    adversary_grads = tape1.gradient(loss_adv, classifier_vars)
                    for _, (grad, var) in enumerate(zip(classifier_grad, self.clf_model.trainable_variables)):
                        unit_adversary_grad = normalize(adversary_grads[_])
                        grad -= tf.reduce_sum(grad * unit_adversary_grad) * unit_adversary_grad
                        grad -= self.adversary_loss_weight * adversary_grads[_]
                        classifier_grads.append((grad, var))
                    classifier_opt.apply_gradients(classifier_grads)
                    with tf.GradientTape() as tape2:
                        pred_protected_attributes_labels, pred_protected_attributes_logits = self.adv_model.forward(
                            pred_logits, batch_labels)
                        loss_adv = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_protected_attributes.reshape(self.batch_size,n_groups),
                                                                    logits=pred_protected_attributes_logits))
                    gradients = tape2.gradient(loss_adv, self.adv_model.trainable_variables)
                    adversary_opt.apply_gradients(zip(gradients, self.adv_model.trainable_variables))
                    if i % 200 == 0:
                        print(
                            "(Adversarial Debiasing) epoch %d; iter: %d; batch classifier loss: %f; batch adversarial loss: %f" % (
                                epoch, i, loss_clf, loss_adv))

        else:
            for epoch in range(self.num_epochs):
                shuffled_ids = [i for i in range(num_train_samples)]
                # BUG: np.random.choice not reproduce same shuffled id every epochs
                #shuffled_ids = np.random.choice(num_train_samples, num_train_samples, replace=False)
                for i in range(num_train_samples // self.batch_size):
                    batch_ids = shuffled_ids[self.batch_size * i: self.batch_size * (i + 1)]
                    batch_features = X.values[batch_ids].astype('float32')
                    batch_labels = np.reshape(y[batch_ids], [-1, 1]).astype('float32')
                    with tf.GradientTape() as tape:
                        pred_labels, pred_logits = self.clf_model.forward(batch_features)
                        loss_clf = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_labels.astype('float32'),
                                                                    logits=pred_logits))
                    gradients = tape.gradient(loss_clf, self.clf_model.trainable_variables)
                    classifier_opt.apply_gradients(zip(gradients, self.clf_model.trainable_variables))
                    if i % 200 == 0:
                        print("(Training Classifier) epoch %d; iter: %d; batch classifier loss: %f" % (
                            epoch, i, loss_clf))
        return self

    def decision_function(self, X):
        """Soft prediction scores.

        Args:
            X (pandas.DataFrame): Test samples.

        Returns:
            numpy.ndarray: Confidence scores per (sample, class) combination. In
            the binary case, confidence score for ``self.classes_[1]`` where >0
            means this class would be predicted.
        """

        num_test_samples = X.shape[0]
        n_classes = len(self.classes_)

        if n_classes == 2:
            n_classes = 1

        self.clf_model.dropout = 0
        samples_covered = 0
        pred_labels_list = []
        while samples_covered < num_test_samples:
            start = samples_covered
            end = samples_covered + self.batch_size
            if end > num_test_samples:
                end = num_test_samples
            batch_ids = np.arange(start, end)
            batch_features = X.values[batch_ids]
            pred_labels, pred_logits = self.clf_model.forward(batch_features.astype("float32"))

            pred_labels_list += pred_labels.numpy().tolist()
            samples_covered += len(batch_features)

        scores = np.array(pred_labels_list, dtype=np.float64).reshape(-1, 1)
        return scores.ravel() if scores.shape[1] == 1 else scores





    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates for all classes are ordered by the label of
        classes.

        Args:
            X (pandas.DataFrame): Test samples.

        Returns:
            numpy.ndarray: Returns the probability of the sample for each class
            in the model, where classes are ordered as they are in
            ``self.classes_``.
        """
        decision = self.decision_function(X)

        if decision.ndim == 1:
            decision_2d = np.c_[np.zeros_like(decision), decision]
        else:
            decision_2d = decision
        return scipy.special.softmax(decision_2d, axis=1)

    def predict(self, X):
        """Predict class labels for the given samples.

        Args:
            X (pandas.DataFrame): Test samples.

        Returns:
            numpy.ndarray: Predicted class label per sample.
        """
        scores = self.decision_function(X)
        if scores.ndim == 1:
            indices = (scores > 0.5).astype(np.int).reshape(-1,1)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]
