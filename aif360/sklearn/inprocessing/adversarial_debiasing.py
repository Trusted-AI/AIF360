import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_is_fitted, check_random_state
import tensorflow as tf

from aif360.sklearn.utils import check_inputs, check_groups


class AdversarialDebiasing(BaseEstimator, ClassifierMixin):
    """Adversarial debiasing is an in-processing technique that learns a
    classifier to maximize prediction accuracy and simultaneously reduce an
    adversary's ability to determine the protected attribute from the
    predictions [#zhang18]_. This approach leads to a fair classifier as the
    predictions cannot carry any group discrimination information that the
    adversary can exploit.

    References:
        .. [#zhang18] B. H. Zhang, B. Lemoine, and M. Mitchell, "Mitigating
           Unwanted Biases with Adversarial Learning," AAAI/ACM Conference on
           Artificial Intelligence, Ethics, and Society, 2018.
    """

    def __init__(self, prot_attr=None, adversary_loss_weight=0.1, num_epochs=50,
                 batch_size=128, classifier_num_hidden_units=200, debias=True,
                 verbose=True, random_state=None):

        self.prot_attr = prot_attr
        self.adversary_loss_weight = adversary_loss_weight
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.classifier_num_hidden_units = classifier_num_hidden_units
        self.debias = debias
        self.verbose = verbose
        self.random_state = random_state

    @property
    def classifier_logits_(self):
        check_is_fitted(self, ['input_ph', 'keep_prob'])
        with tf.variable_scope('classifier_model'):
            W1 = tf.get_variable(
                    'W1', [self.input_ph.shape[1], self.classifier_num_hidden_units],
                    initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.zeros(shape=[self.classifier_num_hidden_units]),
                    name='b1')

            h1 = tf.nn.relu(tf.matmul(self.input_ph, W1) + b1)
            h1 = tf.nn.dropout(h1, keep_prob=self.keep_prob)

            W2 = tf.get_variable(
                    'W2', [self.classifier_num_hidden_units, 1],
                    initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.zeros(shape=[1]), name='b2')

            pred_logits = tf.matmul(h1, W2) + b2

        return pred_logits

    @property
    def adversary_logits_(self):
        """Compute the adversary predictions for the protected attribute."""
        check_is_fitted(self, ['classifier_logits_', 'true_labels_ph'])
        with tf.variable_scope("adversary_model"):
            c = tf.get_variable('c', initializer=tf.constant(1.0))
            s = tf.sigmoid((1 + tf.abs(c)) * self.classifier_logits)

            W2 = tf.get_variable('W2', [3, 1],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.zeros(shape=[1]), name='b2')

            pred_prot_attr_logits = tf.matmul(
                    tf.concat([s, s * self.true_labels_ph, s * (1.0 - self.true_labels_ph)], axis=1),
                    W2) + b2
            # pred_prot_attr_labels = tf.sigmoid(pred_prot_attr_logit)

        return pred_prot_attr_logits

    def _train(self, X, y, groups):


    def fit(self, X, y):
        rng = check_random_state(self.random_state)
        # tf.random.seed(random_state)

        groups, self.prot_attr_ = check_groups(X, self.prot_attr)
        lb = LabelBinarizer()
        y = lb.fit_transform(y)
        # TODO: LabelEncoder for groups
        self.groups_ = np.unique(groups)
        self.classes_ = lb.classes_
        self.sess_ = tf.Session()

        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        n_groups = len(self.groups_)

        with tf.variable_scope('adversarial_debiasing'):
            n_samples, n_features = X.shape
            n_classes = len(self.classes_)
            n_groups = len(self.groups_)

            # Setup placeholders
            self.input_ph = tf.placeholder(tf.float32, shape=[None, n_features])
            self.prot_attr_ph = tf.placeholder(tf.float32, shape=[None, n_groups])
            self.true_labels_ph = tf.placeholder(tf.float32, shape=[None, n_classes])
            self.keep_prob = tf.placeholder(tf.float32)

            # Obtain classifier loss
            loss_fn = (tf.nn.sigmoid_cross_entropy_with_logits if n_classes == 1
                       else tf.nn.softmax_cross_entropy_with_logits)
            # clf_loss = loss_fn(labels=self.true_labels_ph, logits=self.classifier_logits_)
            # clf_loss = tf.reduce_sum(sample_weight * clf_loss) / tf.reduce_sum(sample_weight)
            clf_loss = tf.reduce_mean(loss_fn(labels=self.true_labels_ph,
                                              logits=self.classifier_logits_))

            if self.debias:
                # Obtain adversary loss
                loss_fn = (tf.nn.sigmoid_cross_entropy_with_logits if n_groups == 1
                           else tf.nn.softmax_cross_entropy_with_logits)
                adv_loss = tf.reduce_mean(loss_fn(labels=self.prot_attr_ph,
                                                  logits=self.adversary_logits_))

            # Setup optimizers with learning rates
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = 0.001
            learning_rate = tf.train.exponential_decay(
                    starter_learning_rate, global_step, 1000, 0.96, staircase=True)
            clf_opt = tf.train.AdamOptimizer(learning_rate)
            if self.debias:
                adv_opt = tf.train.AdamOptimizer(learning_rate)

            clf_vars = [var for var in tf.trainable_variables()
                        if 'classifier_model' in var.name]
            if self.debias:
                adv_vars = [var for var in tf.trainable_variables()
                            if 'adversary_model' in var.name]
                # Update classifier parameters
                adv_grads = {var: grad for (grad, var) in
                        adv_opt.compute_gradients(adv_loss, var_list=clf_vars)}

            normalize = lambda x: x / (tf.norm(x) + np.finfo(np.float32).tiny)

            clf_grads = []
            for (grad, var) in clf_opt.compute_gradients(clf_loss, var_list=clf_vars):
                if self.debias:
                    unit_adv_grad = normalize(adv_grads[var])
                    # proj_{adv_grad} clf_grad:
                    grad -= tf.reduce_sum(grad * unit_adv_grad) * unit_adv_grad
                    grad -= self.adversary_loss_weight * adv_grads[var]
                clf_grads.append((grad, var))
            clf_minimizer = clf_opt.apply_gradients(clf_grads, global_step=global_step)

            if self.debias:
                # Update adversary parameters
                adv_minimizer = adv_opt.minimize(adv_loss, var_list=adv_vars,
                                                 global_step=global_step)

            self.sess_.run(tf.global_variables_initializer())
            self.sess_.run(tf.local_variables_initializer())

            # Begin training
            for epoch in range(self.num_epochs):
                # TODO: why rng.choice(n_samples, n_samples)?
                shuffled_ids = rng.shuffle(np.arange(n_samples))
                for i in range(n_samples // self.batch_size):
                    batch_ids = shuffled_ids[self.batch_size * i:self.batch_size * (i+1)]
                    batch_features = X[batch_ids]
                    batch_labels = y[batch_ids]
                    batch_prot_attr = groups[batch_ids]
                    batch_feed_dict = {self.input_ph: batch_features,
                                       self.true_labels_ph: batch_labels,
                                       self.prot_attr_ph: batch_prot_attr,
                                       self.keep_prob: 0.8}
                    if self.debias:
                        _, _, clf_loss_value, adv_loss_value = (
                                self.sess_.run([clf_minimizer, adv_minimizer,
                                                clf_loss, adv_loss],
                                               feed_dict=batch_feed_dict))
                        if i % 200 == 0 and self.verbose:
                            print("epoch {}; iter: {}; batch classifier loss: "
                                  "{}; batch adversarial loss: {}".format(
                                          epoch, i, clf_loss_value,
                                          adv_loss_value))
                    else:
                        _, clf_loss_value = self.sess_.run(
                                [clf_minimizer, clf_loss],
                                feed_dict=batch_feed_dict)
                        if i % 200 == 0 and self.verbose:
                            print("epoch {}; iter: {}; batch classifier loss: "
                                  "{}".format(epoch, i, clf_loss_value))

        return self

    def decision_function(self, X):
        check_is_fitted(self, ['classes_', 'input_ph', 'keep_prob', 'classifier_logits_'])
        n_samples = X.shape[0]
        groups, _ = check_groups(X, self.prot_attr_)

        samples_covered = 0
        scores = np.empty((n_samples, len(self.classes_)))
        while samples_covered < n_samples:
            start = samples_covered
            end = samples_covered + self.batch_size
            if end > n_samples:
                end = n_samples

            batch_ids = np.arange(start, end)
            batch_features = X[batch_ids]
            batch_prot_attr = groups[batch_ids]

            batch_feed_dict = {self.input_ph: batch_features,
                               self.keep_prob: 1.0}

            # batch_logits = self.sess_.run(self.classifier_logits_, feed_dict=batch_feed_dict)
            scores[batch_ids] = self.sess_.run(self.classifier_logits_,
                                               feed_dict=batch_feed_dict)
            samples_covered += len(batch_features)

        return scores

    def predict_proba(self, X):
        decision = self.decision_function(X)
        return softmax(decision, axis=1)

    def predict(self, X):
        indices = self.decision_function(X).argmax(axis=1)
        return self.classes_[indices]
