import numpy as np
import tensorflow as tf
from aif360.algorithms import Transformer

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
    def __init__(self, seed3):
        super(adversary_model, self).__init__()
        self.seed3 = seed3
        self.c = tf.Variable(tf.constant(1.0), name = 'c')
        self.W1 = tf.Variable(tf.random.normal(shape=(3, 1),seed=self.seed3), name='w1')
        self.b1 = tf.Variable(tf.zeros(shape=1), name = 'b1')
        self.sigmoid = tf.nn.sigmoid

    def forward(self,pred_logits, true_labels):
        s = self.sigmoid((1 + tf.abs(self.c)) * pred_logits)
        pred_protected_attribute_logits = tf.matmul(tf.concat([s, s * true_labels, s * (1.0 - true_labels)], axis=1), self.W1) + self.b1
        pred_protected_attribute_labels = tf.sigmoid(pred_protected_attribute_logits)
        return pred_protected_attribute_labels, pred_protected_attribute_logits

class AdversarialDebiasing(Transformer):
    """Adversarial debiasing is an in-processing technique that learns a
    classifier to maximize prediction accuracy and simultaneously reduce an
    adversary's ability to determine the protected attribute from the
    predictions [5]_. This approach leads to a fair classifier as the
    predictions cannot carry any group discrimination information that the
    adversary can exploit.

    References:
        .. [5] B. H. Zhang, B. Lemoine, and M. Mitchell, "Mitigating Unwanted
           Biases with Adversarial Learning," AAAI/ACM Conference on Artificial
           Intelligence, Ethics, and Society, 2018.
    """

    def __init__(self,
                 unprivileged_groups,
                 privileged_groups,
                 scope_name,
                 seed=None,
                 adversary_loss_weight=0.1,
                 num_epochs=50,
                 batch_size=256,
                 classifier_num_hidden_units=200,
                 debias=True):
        """
        Args:
            unprivileged_groups (tuple): Representation for unprivileged groups
            privileged_groups (tuple): Representation for privileged groups
            scope_name (str): scope name for the tenforflow variables
            sess (tf.Session): tensorflow session
            seed (int, optional): Seed to make `predict` repeatable.
            adversary_loss_weight (float, optional): Hyperparameter that chooses
                the strength of the adversarial loss.
            num_epochs (int, optional): Number of training epochs.
            batch_size (int, optional): Batch size.
            classifier_num_hidden_units (int, optional): Number of hidden units
                in the classifier model.
            debias (bool, optional): Learn a classifier with or without
                debiasing.
        """
        super(AdversarialDebiasing, self).__init__(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)

        self.scope_name = scope_name
        self.seed = seed

        self.unprivileged_groups = unprivileged_groups
        self.privileged_groups = privileged_groups
        if len(self.unprivileged_groups) > 1 or len(self.privileged_groups) > 1:
            raise ValueError("Only one unprivileged_group or privileged_group supported.")
        self.protected_attribute_name = list(self.unprivileged_groups[0].keys())[0]
        self.adversary_loss_weight = adversary_loss_weight
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.classifier_num_hidden_units = classifier_num_hidden_units
        self.debias = debias
        self.features_dim = None
        self.features_ph = None
        self.protected_attributes_ph = None
        self.true_labels_ph = None
        self.pred_labels = None



    def fit(self, dataset):
        """Compute the model parameters of the fair classifier using gradient
        descent.

        Args:
            dataset (BinaryLabelDataset): Dataset containing true labels.

        Returns:
            AdversarialDebiasing: Returns self.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        ii32 = np.iinfo(np.int32)
        self.seed1, self.seed2, self.seed3, self.seed4 = np.random.randint(ii32.min, ii32.max, size=4)

        # Map the dataset labels to 0 and 1.
        temp_labels = dataset.labels.copy()

        temp_labels[(dataset.labels == dataset.favorable_label).ravel(),0] = 1.0
        temp_labels[(dataset.labels == dataset.unfavorable_label).ravel(),0] = 0.0
        num_train_samples, self.features_dim = np.shape(dataset.features)
        starter_learning_rate = 0.001
        self.clf_model = classifier_model(feature=self.features_dim, Hneuron1=self.classifier_num_hidden_units, output=1, dropout=0.2,
                                     seed1=self.seed1, seed2=self.seed2)
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(starter_learning_rate,
                                                   decay_steps = 1000, decay_rate=0.96, staircase=True)
        classifier_opt = tf.optimizers.Adam(learning_rate)
        classifier_vars = [var for var in self.clf_model.trainable_variables]

        #pretrain_both_models
        if self.debias:
            self.adv_model = adversary_model(seed3=self.seed3)
            learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(starter_learning_rate,
                                                                           decay_steps=1000, decay_rate=0.96,
                                                                           staircase=True)
            adversary_opt = tf.optimizers.Adam(learning_rate)
            #adversary_vars = [var for var in self.adv_model.trainable_variables]
            for epoch in range(self.num_epochs):
                shuffled_ids = np.random.choice(num_train_samples, num_train_samples, replace=False)
                for i in range(num_train_samples // self.batch_size):
                    batch_ids = shuffled_ids[self.batch_size * i: self.batch_size * (i + 1)]
                    batch_features = dataset.features[batch_ids].astype('float32')
                    batch_labels = np.reshape(temp_labels[batch_ids], [-1, 1]).astype('float32')
                    with tf.GradientTape() as tape:
                        pred_labels, pred_logits = self.clf_model.forward(batch_features)
                        loss_clf = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_labels, logits=pred_logits))
                    gradients = tape.gradient(loss_clf, self.clf_model.trainable_variables)
                    classifier_opt.apply_gradients(zip(gradients,self.clf_model.trainable_variables))
                    if i % 200 == 0:
                        print("(Pretraining Classifier) epoch %d; iter: %d; batch classifier loss: %f" % (
                            epoch, i, loss_clf))
            for epoch in range(self.num_epochs//5):
                shuffled_ids = np.random.choice(num_train_samples, num_train_samples, replace=False)
                for i in range(num_train_samples // self.batch_size):
                    batch_ids = shuffled_ids[self.batch_size * i: self.batch_size * (i + 1)]
                    batch_features = dataset.features[batch_ids].astype('float32')
                    batch_labels = np.reshape(temp_labels[batch_ids], [-1, 1]).astype('float32')
                    batch_protected_attributes = np.reshape(dataset.protected_attributes[batch_ids][:,
                                                            dataset.protected_attribute_names.index(
                                                                self.protected_attribute_name)], [-1, 1]).astype('float32')
                    with tf.GradientTape() as tape:
                        pred_labels, pred_logits = self.clf_model.forward(batch_features)
                        pred_protected_attributes_labels, pred_protected_attributes_logits = self.adv_model.forward(pred_logits, batch_labels)
                        loss_adv = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_protected_attributes, logits=pred_protected_attributes_logits))
                    gradients = tape.gradient(loss_adv, self.adv_model.trainable_variables)
                    adversary_opt.apply_gradients(zip(gradients, self.adv_model.trainable_variables))
                    if i % 200 == 0:
                        print("(Pretraining Adversarial Net) epoch %d; iter: %d; batch classifier loss: %f" % (
                            epoch, i, loss_clf))
            #Adversary Debiasing
            normalize = lambda  x: x / (tf.norm(x) + np.finfo(np.float32).tiny)
            for epoch in range(self.num_epochs):
                #self.clf_model.dropout=0
                #self.adversary_loss_weight=sqrt(epoch)
                shuffled_ids = np.random.choice(num_train_samples, num_train_samples, replace=False)
                for i in range(num_train_samples // self.batch_size):
                    batch_ids = shuffled_ids[self.batch_size * i: self.batch_size * (i + 1)]
                    batch_features = dataset.features[batch_ids].astype('float32')
                    batch_labels = np.reshape(temp_labels[batch_ids], [-1, 1]).astype('float32')
                    batch_protected_attributes = np.reshape(dataset.protected_attributes[batch_ids][:,
                                                            dataset.protected_attribute_names.index(
                                                                self.protected_attribute_name)], [-1, 1]).astype('float32')
                    with tf.GradientTape() as tape:
                        pred_labels, pred_logits = self.clf_model.forward(batch_features)
                        loss_clf =  tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_labels, logits=pred_logits))
                    classifier_grad = tape.gradient(loss_clf,classifier_vars)
                    classifier_grads = []

                    with tf.GradientTape() as tape1:
                        pred_labels, pred_logits = self.clf_model.forward(batch_features) #varaibles of CLF_model need to be watched from tape1 also
                        pred_protected_attributes_labels, pred_protected_attributes_logits = self.adv_model.forward(
                            pred_logits, batch_labels)
                        loss_adv = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_protected_attributes,
                                                                    logits=pred_protected_attributes_logits))
                    adversary_grads = tape1.gradient(loss_adv, classifier_vars)
                    for _, (grad,var) in enumerate(zip(classifier_grad,self.clf_model.trainable_variables)):
                        unit_adversary_grad = normalize(adversary_grads[_])
                        grad -= tf.reduce_sum(grad * unit_adversary_grad) * unit_adversary_grad
                        grad -= self.adversary_loss_weight * adversary_grads[_]
                        classifier_grads.append((grad,var))
                    classifier_opt.apply_gradients(classifier_grads)
                    with tf.GradientTape() as tape2:
                        pred_protected_attributes_labels, pred_protected_attributes_logits = self.adv_model.forward(
                            pred_logits, batch_labels)
                        loss_adv = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_protected_attributes,
                                                                    logits=pred_protected_attributes_logits))
                    gradients = tape2.gradient(loss_adv, self.adv_model.trainable_variables)
                    adversary_opt.apply_gradients(zip(gradients, self.adv_model.trainable_variables))
                    if i % 200 == 0:
                        print("(Adversarial Debiasing) epoch %d; iter: %d; batch classifier loss: %f; batch adversarial loss: %f" % (
                            epoch, i, loss_clf, loss_adv))

        else:
            for epoch in range(self.num_epochs):
                shuffled_ids = np.random.choice(num_train_samples, num_train_samples, replace=False)
                for i in range(num_train_samples // self.batch_size):
                    batch_ids = shuffled_ids[self.batch_size * i: self.batch_size * (i + 1)]
                    batch_features = dataset.features[batch_ids].astype('float32')
                    batch_labels = np.reshape(temp_labels[batch_ids], [-1, 1]).astype('float32')
                    with tf.GradientTape() as tape:
                        pred_labels, pred_logits = self.clf_model.forward(batch_features)
                        loss_clf = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_labels.astype('float32'), logits=pred_logits))
                    gradients = tape.gradient(loss_clf, self.clf_model.trainable_variables)
                    classifier_opt.apply_gradients(zip(gradients, self.clf_model.trainable_variables))
                    if i % 200 == 0:
                        print("(Training Classifier) epoch %d; iter: %d; batch classifier loss: %f" % (
                            epoch, i, loss_clf))
        return self

    def predict(self, dataset):
        """Obtain the predictions for the provided dataset using the fair
        classifier learned.

        Args:
            dataset (BinaryLabelDataset): Dataset containing labels that needs
                to be transformed.
        Returns:
            dataset (BinaryLabelDataset): Transformed dataset.
        """

        if self.seed is not None:
            np.random.seed(self.seed)

        num_test_samples, _ = np.shape(dataset.features)
        self.clf_model.dropout=0
        samples_covered = 0
        pred_labels_list = []
        while samples_covered < num_test_samples:
            start = samples_covered
            end = samples_covered + self.batch_size
            if end > num_test_samples:
                end = num_test_samples
            batch_ids = np.arange(start, end)
            batch_features = dataset.features[batch_ids]
            pred_labels, pred_logits = self.clf_model.forward(batch_features.astype("float32"))

            pred_labels_list += pred_labels.numpy().tolist()
            samples_covered += len(batch_features)

        # Mutated, fairer dataset with new labels
        dataset_new = dataset.copy(deepcopy = True)
        dataset_new.scores = np.array(pred_labels_list, dtype=np.float64).reshape(-1, 1)
        dataset_new.labels = (np.array(pred_labels_list)>0.5).astype(np.float64).reshape(-1,1)


        # Map the dataset labels to back to their original values.
        temp_labels = dataset_new.labels.copy()

        temp_labels[(dataset_new.labels == 1.0).ravel(), 0] = dataset.favorable_label
        temp_labels[(dataset_new.labels == 0.0).ravel(), 0] = dataset.unfavorable_label

        dataset_new.labels = temp_labels.copy()

        return dataset_new
