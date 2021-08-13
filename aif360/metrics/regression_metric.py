from sklearn.metrics import mean_absolute_error
import numpy as np

def mae(dataset, dataset_pred):
    # Get the sensitive attribute
    sensitive_attribute = dataset.protected_attribute_names[0]
    # Get index of the first sensitive attribute
    sensitive_index = dataset.protected_attribute_names.index(sensitive_attribute)
    # Get index of prediction for priviledged group
    label_index = np.where(dataset.features[:,sensitive_index] == dataset.privileged_protected_attributes[0][0])
    # Extract priviledged group true and predicted values using index
    return mean_absolute_error(dataset.labels[label_index[0]], dataset_pred.labels[label_index[0]])