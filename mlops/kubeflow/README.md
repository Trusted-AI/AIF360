# Kubeflow Pipeline Components for AIF360

Kubeflow pipeline components are implementations of Kubeflow pipeline tasks. Each task takes
one or more [artifacts](https://www.kubeflow.org/docs/pipelines/overview/concepts/output-artifact/)
as input and may produce one or more
[artifacts](https://www.kubeflow.org/docs/pipelines/overview/concepts/output-artifact/) as output.


**Example: AIF360 Components**
* [Bias Detector - PyTorch](bias_detector_pytorch)

Each task usually includes two parts:

Each component has a component.yaml which will describe the finctionality exposed by it, for e.g.

```
name: 'PyTorch Model Fairness Check'
description: |
  Perform a fairness check on a certain attribute using AIF360 to make sure the PyTorch model is fair
metadata:
  annotations: {platform: 'OpenSource'}
inputs:
  - {name: model_id,                     description: 'Required. Training model ID', default: 'training-dummy'}
  - {name: model_class_file,             description: 'Required. pytorch model class file'}
  - {name: model_class_name,             description: 'Required. pytorch model class name', default: 'model'}
  - {name: feature_testset_path,         description: 'Required. Feature test dataset path in the data bucket'}
  - {name: label_testset_path,           description: 'Required. processed_data/y_test.npy'}
  - {name: protected_label_testset_path, description: 'Required. Protected label test dataset path in the data bucket'}
  - {name: favorable_label,              description: 'Required. Favorable label for this model predictions'}
  - {name: unfavorable_label,            description: 'Required. Unfavorable label for this model predictions'}
  - {name: privileged_groups,            description: 'Required. Privileged feature groups within this model'}
  - {name: unprivileged_groups,          description: 'Required. Unprivileged feature groups within this model'}
outputs:
  - {name: metric_path,                  description: 'Path for fairness check output'}
implementation:
  container:
    image: aipipeline/fairness-check-with-secret:pytorch-v3
    command: ['python']
    args: [
      -u, fairness_check.py,
      --model_id, {inputValue: model_id},
      --model_class_file, {inputValue: model_class_file},
      --model_class_name, {inputValue: model_class_name},
      --feature_testset_path, {inputValue: feature_testset_path},
      --label_testset_path, {inputValue: label_testset_path},
      --protected_label_testset_path, {inputValue: protected_label_testset_path},
      --favorable_label, {inputValue: favorable_label},
      --unfavorable_label, {inputValue: unfavorable_label},
      --privileged_groups, {inputValue: privileged_groups},
      --unprivileged_groups, {inputValue: unprivileged_groups},
      --metric_path, {outputPath: metric_path}
    ]
```

See how to [use the Kubeflow Pipelines SDK](https://www.kubeflow.org/docs/pipelines/sdk/sdk-overview/)
and [build your own components](https://www.kubeflow.org/docs/pipelines/sdk/build-component/).