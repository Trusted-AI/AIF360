from aif360 import lazy_import

lazy_import(__name__,
            'aif360.algorithms.preprocessing.disparate_impact_remover',
           ['DisparateImpactRemover'])
lazy_import(__name__, 'aif360.algorithms.preprocessing.lfr', ['LFR'])
lazy_import(__name__, 'aif360.algorithms.preprocessing.optim_preproc',
           ['OptimPreproc'])
from aif360.algorithms.preprocessing.reweighing import Reweighing
