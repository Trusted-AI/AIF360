# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2023 Fujitsu Limited
#
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

from abc import ABCMeta
from abc import abstractmethod


class InProcessing(metaclass=ABCMeta):
    """
    Abstract Base Class for all inprocessing techniques.
    """
    def __init__(self):
        super().__init__()
        self.model = None

    @abstractmethod
    def fit(self, ds_train):
        """
        Train a model on the input.

        Parameters
        ----------
        ds_train : Dataset
            Training Dataset.
        """
        pass

    @abstractmethod
    def predict(self, ds):
        """
        Predict on the input.

        Parameters
        ----------
        ds : Dataset
            Dataset to predict.
        """
        pass
