import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from aif360.datasets import StructuredDataset
from aif360.metrics import SampleDistortionMetric


data = np.arange(12).reshape((3, 4)).T
cols = ['one', 'two', 'three', 'label']
labs = np.ones((4, 1))

df = pd.DataFrame(data=np.concatenate((data, labs), axis=1), columns=cols)
sd = StructuredDataset(df=df, label_names=['label'],
    protected_attribute_names=['one', 'three'])

distorted = data + 1

sd_distorted = sd.copy(True)
sd_distorted.features = distorted

rand = np.random.randint(0, 10, (4, 4))
rand2 = np.random.randint(0, 10, (4, 3))
df_rand = pd.DataFrame(data=rand, columns=cols)
sd_rand = StructuredDataset(df=df_rand, label_names=['label'],
    protected_attribute_names=['one', 'three'])
sd_rand2 = sd_rand.copy(True)
sd_rand2.features = rand2


priv = [{'one': 1}]
unpriv = [{'one': 2}]

def test_euclidean_distance():
    sdm = SampleDistortionMetric(sd, sd_distorted)
    assert sdm.total_euclidean_distance() == 4*np.sqrt(3)

def test_manhattan_distance():
    sdm = SampleDistortionMetric(sd, sd_distorted)
    assert sdm.total_manhattan_distance() == 12

def test_mahalanobis_distance():
    sdm = SampleDistortionMetric(sd_rand, sd_rand2)
    assert np.isclose(sdm.total_mahalanobis_distance(),
        np.diag(cdist(rand[:, :3], rand2[:, :3], 'mahalanobis')).sum())

def test_conditional():
    sdm = SampleDistortionMetric(sd, sd_distorted, unprivileged_groups=unpriv,
        privileged_groups=priv)
    assert sdm.total_manhattan_distance(privileged=False) == 3

def test_average():
    sd_distorted.features[-1, -1] += 1
    sd.instance_weights = sd_distorted.instance_weights = np.array([1, 1, 1, 3])
    sdm = SampleDistortionMetric(sd, sd_distorted)
    assert sdm.average_manhattan_distance() == 3.5

def test_error():
    try:
        sd.protected_attributes -= 1
        sdm = SampleDistortionMetric(sd, sd_distorted)
    except ValueError:
        assert True
