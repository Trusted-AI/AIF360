# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from itertools import batched
import numpy as np
import pandas as pd
from minio import Minio
import re

import onnxruntime as ort


def dataset_wrapper(outcome, protected, unprivileged_groups, privileged_groups, favorable_label, unfavorable_label):
    """ A wrapper function to create aif360 dataset from outcome and protected in numpy array format.
    """
    df = pd.DataFrame(data=outcome,
                      columns=['outcome'])
    df['race'] = protected

    dataset = BinaryLabelDataset(favorable_label=favorable_label,
                                 unfavorable_label=unfavorable_label,
                                 df=df,
                                 label_names=['outcome'],
                                 protected_attribute_names=['race'],
                                 unprivileged_protected_attributes=unprivileged_groups)
    return dataset

# Compute the accuaracy and predicted label using the given test dataset
def evaluate(model, X_test, y_test):
    input_name = model.get_inputs()[0].name
    test = X_test.astype('float32')

    y_pred = []
    for images in batched(test, 64):
        outputs = model.run(None, {input_name: images})
        predictions = np.argmax(outputs[0], axis=1)
        y_pred.append(predictions)

    y_pred = np.concatenate(y_pred, axis=0)
    accuracy = np.mean(y_pred == y_test)

    return accuracy, y_pred


def fairness_check(object_storage_url, object_storage_username, object_storage_password,
                   data_bucket_name, result_bucket_name, model_id,
                   feature_testset_path='processed_data/X_test.npy',
                   label_testset_path='processed_data/y_test.npy',
                   protected_label_testset_path='processed_data/p_test.npy',
                   model_file='model.onnx',
                   favorable_label=0.0,
                   unfavorable_label=1.0,
                   privileged_groups=[{'race': 0.0}],
                   unprivileged_groups=[{'race': 4.0}]):

    url = re.compile(r"https?://")
    cos = Minio(url.sub('', object_storage_url),
                access_key=object_storage_username,
                secret_key=object_storage_password,
                secure=False)  # Local Minio server won't have HTTPS

    dataset_filenamex = "X_test.npy"
    dataset_filenamey = "y_test.npy"
    dataset_filenamep = "p_test.npy"
    weights_filename = "model.pt"

    cos.fget_object(data_bucket_name, feature_testset_path, dataset_filenamex)
    cos.fget_object(data_bucket_name, label_testset_path, dataset_filenamey)
    cos.fget_object(data_bucket_name, protected_label_testset_path, dataset_filenamep)
    cos.fget_object(result_bucket_name, model_id + '/' + weights_filename, weights_filename)
    cos.fget_object(result_bucket_name, model_id + '/' + model_file, model_file)

    # Load ONNX model with security settings
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Create inference session
    model = ort.InferenceSession(
        model_file,
        sess_options=sess_options,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    """Load the necessary labels and protected features for fairness check"""

    x_test = np.load(dataset_filenamex)
    y_test = np.load(dataset_filenamey)
    p_test = np.load(dataset_filenamep)

    _, y_pred = evaluate(model, x_test, y_test)

    """Calculate the fairness metrics"""

    original_test_dataset = dataset_wrapper(outcome=y_test, protected=p_test,
                                            unprivileged_groups=unprivileged_groups,
                                            privileged_groups=privileged_groups,
                                            favorable_label=favorable_label,
                                            unfavorable_label=unfavorable_label)
    plain_predictions_test_dataset = dataset_wrapper(outcome=y_pred, protected=p_test,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups,
                                                     favorable_label=favorable_label,
                                                     unfavorable_label=unfavorable_label)

    classified_metric_nodebiasing_test = ClassificationMetric(original_test_dataset,
                                                              plain_predictions_test_dataset,
                                                              unprivileged_groups=unprivileged_groups,
                                                              privileged_groups=privileged_groups)
    TPR = classified_metric_nodebiasing_test.true_positive_rate()
    TNR = classified_metric_nodebiasing_test.true_negative_rate()
    bal_acc_nodebiasing_test = 0.5*(TPR+TNR)

    print("#### Plain model - without debiasing - classification metrics on test set")

    metrics = {
        "Classification accuracy": classified_metric_nodebiasing_test.accuracy(),
        "Balanced classification accuracy": bal_acc_nodebiasing_test,
        "Statistical parity difference": classified_metric_nodebiasing_test.statistical_parity_difference(),
        "Disparate impact": classified_metric_nodebiasing_test.disparate_impact(),
        "Equal opportunity difference": classified_metric_nodebiasing_test.equal_opportunity_difference(),
        "Average odds difference": classified_metric_nodebiasing_test.average_odds_difference(),
        "Theil index": classified_metric_nodebiasing_test.theil_index(),
        "False negative rate difference": classified_metric_nodebiasing_test.false_negative_rate_difference()
    }
    print("metrics: ", metrics)
    return metrics
