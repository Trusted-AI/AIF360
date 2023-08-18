import tensorflow as tf
import os
import numpy as np
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from tensorflow.keras.layers import Dense, Dropout, Activation, Concatenate
from tensorflow.keras.initializers import glorot_uniform



class ClassifierModel(tf.keras.Model):
    def __init__(self, classifier_num_hidden_units=256, seed=None):
        super(ClassifierModel, self).__init__(name='classifier')

        self.dense_layer_1 = Dense(classifier_num_hidden_units,
                                   activation='relu',
                                   kernel_initializer=glorot_uniform(seed=seed))
        self.dropout_layer = Dropout(0.5)
        self.dense_layer_2 = Dense(classifier_num_hidden_units, activation='relu',
                                   kernel_initializer=glorot_uniform(seed=seed))
        self.output_logit_layer = Dense(1, activation=None)
        self.output_label_activation = Activation('sigmoid')
        
    def call(self, features):
        x = self.dense_layer_1(features)
        x = self.dropout_layer(x)
        x = self.dense_layer_2(x)
        pred_logit = self.output_logit_layer(x)
        pred_label = self.output_label_activation(pred_logit)
        return pred_logit



class AdversaryModel(tf.keras.Model):
    def __init__(self, seed=None, nb_sensitive_features=2):
        super(AdversaryModel, self).__init__(name='adversary')
        self.c = self.add_weight('c', shape=(), initializer='ones')
        self.dense_layer = Dense(64,
                                 activation='relu',
                                 kernel_initializer=glorot_uniform(seed=seed))
        self.output_logit_layer = Dense(nb_sensitive_features, activation=None)
        self.output_label_activation = Activation('sigmoid')
        
    def call(self, pred_logits, true_labels):
        s = tf.keras.activations.sigmoid((1 + tf.abs(self.c)) * pred_logits)
        x = Concatenate(axis=1)([s, s * true_labels, s * (1.0 - true_labels)])
        x = self.dense_layer(x)
        pred_protected_attribute_logits = self.output_logit_layer(x)
        pred_protected_attribute_labels = self.output_label_activation(pred_protected_attribute_logits)
        return pred_protected_attribute_labels, pred_protected_attribute_logits




class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    '''Loss function for adversary. Weights can be provided for each sensitive attribute.'''
    def __init__(self, weights, **kwargs):
        super(WeightedBinaryCrossEntropy, self).__init__(**kwargs)
        self.weights = weights

    def call(self, y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        losses = []
        for i, weight in enumerate(self.weights):
            loss = -tf.reduce_mean(y_true[:, i] * tf.math.log(y_pred[:, i]) + (1.0 - y_true[:, i]) * tf.math.log(1.0 - y_pred[:, i]))
            losses.append(weight * loss)
        return tf.reduce_sum(losses)



class AdversarialDebiasor:
    '''TODO: 
    
    # Make train_step() faster
    # 2. Make different types of adversaries.
        
        Depending on the definition of fairness being achieved, the adversary may have other inputs.

            • For DEMOGRAPHIC PARITY, the adversary gets the predicted label Yˆ . Intuitively, this allows the adversary to try
              to predict the protected variable using nothing but the predicted label. The goal of the predictor is to prevent the
              adversary from doing this.

            • For EQUALITY OF ODDS, the adversary gets Yˆ and the true label Y .

            • For EQUALITY OF OPPORTUNITY on a given class y, we can restrict the training set of the adversary to training examples where Y = y.
 
       References:
            [1] B. H. Zhang, B. Lemoine, and M. Mitchell, "Mitigating UnwantedBiases with Adversarial Learning," 
            AAAI/ACM Conference on Artificial Intelligence, Ethics, and Society, 2018.
 
 '''
    def __init__(self, classifier=ClassifierModel(),
                  adversary=AdversaryModel(),
                    loss_weights = [.5, .5],
                      classifier_optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                        adversary_optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)):
        self.classifier = classifier
        self.adversary = adversary
        self.loss_fn = WeightedBinaryCrossEntropy(weights=loss_weights)
        self.classifier_optimizer = classifier_optimizer
        self.adversary_optimizer = adversary_optimizer
        self.debias = True
        self.adversary_loss_weight = 0.1
        if 'tensorflow_privacy' in str(classifier_optimizer):
            self.differential_privacy=True
        else:
            self.differential_privacy=False
        

    def pretrain_classifier(self, features, labels, num_epochs, batch_size):
        ''''''
        num_samples = features.shape[0]
        for epoch in range(num_epochs):
            permuted_indices = np.random.permutation(num_samples)
            for i in range(0, num_samples, batch_size):
                batch_indices = permuted_indices[i:i+batch_size]
                batch_features = tf.convert_to_tensor(features[batch_indices], dtype=tf.float32)
                batch_labels = tf.convert_to_tensor(labels[batch_indices], dtype=tf.float32)

                def loss_fn():
                    pred_logits = self.classifier(batch_features)
                    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_labels, logits=pred_logits))
                
                # Use gradient tape to compute the gradients
                with tf.GradientTape() as tape:
                    loss_value = loss_fn()
                    # Use compute_gradients with a callable loss function and pass the gradient tape
                    if self.differential_privacy:
                        grads_and_vars = self.classifier_optimizer.compute_gradients(loss_value, self.classifier.trainable_variables, gradient_tape=tape)
                        self.classifier_optimizer.apply_gradients(grads_and_vars)
                    else:
                        grads = tape.gradient(loss_value, self.classifier.trainable_variables)
                        self.classifier_optimizer.apply_gradients(zip(grads, self.classifier.trainable_variables))
                    
            
            print(f"Pretraining Classifier - Epoch {epoch+1}, Loss: {loss_fn().numpy()}")




    def pretrain_adversary(self, features, labels, protected_attributes, num_epochs, batch_size):
        ''''''
        num_samples = features.shape[0]
        for epoch in range(num_epochs):
            permuted_indices = np.random.permutation(num_samples)
            for i in range(0, num_samples, batch_size):
                batch_indices = permuted_indices[i:i+batch_size]
                batch_features = tf.convert_to_tensor(features[batch_indices], dtype=tf.float32)
                batch_labels = tf.convert_to_tensor(labels[batch_indices], dtype=tf.float32)
                batch_protected_attributes = tf.convert_to_tensor(protected_attributes[batch_indices], dtype=tf.float32)
                
                pred_logits = self.classifier(batch_features)
                
                with tf.GradientTape() as tape:
                    pred_protected_attribute_labels, pred_protected_attribute_logits = self.adversary(pred_logits, batch_labels)
                    adversary_loss = self.loss_fn(batch_protected_attributes, pred_protected_attribute_labels)
                grads = tape.gradient(adversary_loss, self.adversary.trainable_variables)
                self.adversary_optimizer.apply_gradients(zip(grads, self.adversary.trainable_variables))
            
            print(f"Pretraining Adversary - Epoch {epoch+1}, Loss: {adversary_loss.numpy()}")


    
    def train_step(self, batch_features, batch_labels, batch_protected_attributes):
        ''' INFO: Training step for the model'''

        # Create a GradientTape to automatically track operations and compute gradients
        with tf.GradientTape(persistent=True) as tape:
            # Get the predicted logits from the classifier
            pred_logits = self.classifier(batch_features)
            # Compute the classifier loss using sigmoid cross-entropy
            classifier_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_labels, logits=pred_logits))
            # If debiasing is enabled, compute the adversary loss
            if self.debias:
                # Get the predicted labels and logits for the protected attributes from the adversary
                pred_protected_attribute_labels, pred_protected_attribute_logits = self.adversary(pred_logits, batch_labels)
                # Compute the adversary loss using the specified loss function
                adversary_loss = self.loss_fn(batch_protected_attributes, pred_protected_attribute_labels)

        # Get the list of trainable variables in the classifier
        classifier_vars = [var for var in self.classifier.trainable_variables]

        # If debiasing is enabled, perform additional computations
        if self.debias:
            # Get the list of trainable variables in the adversary
            adversary_vars = [var for var in self.adversary.trainable_variables]
            # Compute the gradients of the adversary loss with respect to the classifier variables
            adversary_grads = {var.name: grad for (grad, var) in zip(tape.gradient(adversary_loss, classifier_vars), classifier_vars)}

            # Define a normalization function to normalize a tensor
            normalize = lambda x: x / (tf.norm(x) + np.finfo(np.float32).tiny)

            # Initialize a list to store the updated classifier gradients
            updated_classifier_grads = []
            # For each gradient and variable in the classifier
            for (grad, var) in zip(tape.gradient(classifier_loss, classifier_vars), classifier_vars):
                # If the variable is present in the adversary gradients
                if var.name in adversary_grads:
                    # Normalize the corresponding adversary gradient
                    unit_adversary_grad = normalize(adversary_grads[var.name])
                    # Subtract the projection of the classifier gradient onto the normalized adversary gradient
                    grad -= tf.reduce_sum(grad * unit_adversary_grad) * unit_adversary_grad
                    # Subtract the weighted adversary gradient
                    grad -= self.adversary_loss_weight * adversary_grads[var.name]
                # Append the updated gradient and variable to the list
                updated_classifier_grads.append((grad, var))

            # Apply the updated gradients to the classifier variables
            self.classifier_optimizer.apply_gradients(updated_classifier_grads)

            # If debiasing is enabled, update the adversary variables
            if self.debias:
                # Ensure the classifier is updated before the adversary
                with tf.control_dependencies([self.classifier_optimizer.apply_gradients(updated_classifier_grads)]):
                    # Minimize the adversary loss with respect to the adversary variables
                    self.adversary_optimizer.minimize(adversary_loss, var_list=adversary_vars, tape=tape)

        # Return the classifier and adversary losses
        return classifier_loss, adversary_loss




    def train(self, features, labels, protected_attributes, num_epochs, batch_size):
        ''''''
        num_samples = features.shape[0]
        for epoch in range(num_epochs):
            permuted_indices = np.random.permutation(num_samples)
            for i in range(0, num_samples, batch_size):
                batch_indices = permuted_indices[i:i+batch_size]
                batch_features = tf.convert_to_tensor(features[batch_indices], dtype=tf.float32)
                batch_labels = tf.convert_to_tensor(labels[batch_indices], dtype=tf.float32)
                batch_protected_attributes = tf.convert_to_tensor(protected_attributes[batch_indices], dtype=tf.float32)
                
                classifier_loss, adversary_loss = self.train_step(batch_features, batch_labels, batch_protected_attributes)
            
            print(f"Epoch {epoch+1}, Classifier Loss: {classifier_loss.numpy()}, Adversary Loss: {adversary_loss.numpy()}")


    def predict_proba(self, X):
        ''''''
        logits = self.classifier(X)
        return tf.nn.sigmoid(logits)
    
    def predict(self, X, threshold=.5):
        ''''''
        return tf.cast(self.predict_proba(X) >= threshold, dtype=tf.int32)
    
    def transform_dataset(self, dataset):
        ''''''
        preds = self.predict_proba(dataset.features)
        # Mutated, fairer dataset with new labels
        dataset_new = dataset.copy(deepcopy = True)
        dataset_new.scores = np.array(preds, dtype=np.float64).reshape(-1, 1)
        dataset_new.labels = (np.array(preds)>0.5).astype(np.float64).reshape(-1,1)

        return dataset_new
    
    
    def get_dataset_metrics(self, dataset):
        ''''''
        metrics = {s:'' for s in dataset.protected_attribute_names}
        for attr in dataset.protected_attribute_names:
            unprivileged_group, privileged_group = {attr:0}, {attr:1}
            new_dataset = self.transform_dataset(dataset)
            metrics[attr] = BinaryLabelDatasetMetric(new_dataset, 
                                                     unprivileged_groups=[unprivileged_group],
                                                     privileged_groups=[privileged_group])
        return metrics

    def get_classification_metrics(self, dataset):
        ''''''
        metrics = {s:'' for s in dataset.protected_attribute_names}
        for attr in dataset.protected_attribute_names:
            unprivileged_group, privileged_group = {attr:0}, {attr:1}
            new_dataset = self.transform_dataset(dataset)
            metrics[attr] = ClassificationMetric(dataset, 
                                                 new_dataset,
                                                 unprivileged_groups=[unprivileged_group],
                                                 privileged_groups=[privileged_group])
        return metrics
