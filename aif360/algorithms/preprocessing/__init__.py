from aif360 import lazy_import

# normal imports need to be done before lazy ones
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc

lazy_import(__name__,
            'aif360.algorithms.preprocessing.disparate_impact_remover',
           ['DisparateImpactRemover'])
lazy_import(__name__, 'aif360.algorithms.preprocessing.lfr', ['LFR'])
# lazy_import(__name__, 'aif360.algorithms.preprocessing.optim_preproc',
#            ['OptimPreproc'])
