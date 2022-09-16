import abc
import os

from aif360.stores.store import Store
from aif360.utils import get_logger

# To use Kaggle store,
# 1. Login to Kaggle
# 2. Create new API token
# 3. Save Kaggle.json to ~/.kaggle
# 4. Use KaggleStore


class KaggleStore(Store):

    def __init__(self, source, destination):

        self.logger = get_logger(type(self).__name__)
        self.validate_store(source)
        self.source = source
        self.destination = destination

    def validate_store(self, dataset):
        # TODO: check if dataset actually exists in kaggle?
        pass

    def download(self):
        # importing kaggle inside download() to avoid an authentication error.
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(self.source, path=self.destination,
                                          unzip=True)
        return len(os.listdir(self.destination)) != 0

    def existsInDestination(self, **kwargs):
        pass
