from aif360 import lazy_import

# lazy import AdversarialDebiasing since it requires tensorflow
lazy_import(__name__, 'aif360.algorithms.inprocessing.adversarial_debiasing',
           ['AdversarialDebiasing'])
from aif360.algorithms.inprocessing.art_classifier import ARTClassifier
from aif360.algorithms.inprocessing.prejudice_remover import PrejudiceRemover
from aif360.algorithms.inprocessing.meta_fair_classifier import MetaFairClassifier
