import abc
import os
import urllib.request
from aif360.stores.store import Store
from aif360.utils import get_logger


class WebStore(Store):
    def __init__(self, source, destination, **kwargs):

        self.logger = get_logger(type(self).__name__)
        self.validate_store(source)
        self.source = source
        self.destination = destination

        os.makedirs(self.destination, exist_ok=True)
        self.files_to_download = kwargs["files"]

    def validate_store(self, dataset):
        # TODO: check if dataset actually exists in kaggle?
        pass

    def download(self):
        if not self.existsInDestination(self.files_to_download):
            for data_file in self.files_to_download:
                _ = urllib.request.urlretrieve(os.path.join(self.source, data_file),
                                               os.path.join(self.destination, data_file))
        self.logger.info("DONE")
        return len(os.listdir(self.destination)) != 0

    def existsInDestination(self, files):
        check = all(item in os.listdir(self.destination) for item in files)
        if check:
            self.logger.info("Adult dataset is available in " + self.destination)
        else:
            self.logger.info("Some files are missing. Downloading now.")
            for data_file in files:
                _ = urllib.request.urlretrieve(os.path.join(self.source, data_file),
                                               os.path.join(self.destination, data_file))

# Usage
# kwargs = {"files": ["adult.data", "adult.test", "adult.names"]}
# obj = WebStore(source="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/", destination="output",
#                **kwargs)
# obj.download()
