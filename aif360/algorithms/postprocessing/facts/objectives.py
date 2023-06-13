from pandas import DataFrame

from .recourse_sets import TwoLevelRecourseSet
from .models import ModelAPI
from .metrics import incorrectRecourses, cover, featureCost, featureChange

from .parameters import epsilon1, epsilon2, C_max, M_max


def reward1(R: TwoLevelRecourseSet, X_aff: DataFrame, model: ModelAPI) -> float:
    """Calculate the reward for recourse set R based on the number of correct recourses.

    Args:
        R (TwoLevelRecourseSet): The recourse set.
        X_aff (DataFrame): The affected data instances.
        model (ModelAPI): The machine learning model.

    Returns:
        float: The reward value.
    """
    U_1 = len(X_aff) * epsilon1
    return U_1 - incorrectRecourses(R, X_aff, model)


reward2 = cover


def reward3(R: TwoLevelRecourseSet) -> float:
    """Calculate the reward for recourse set R based on the feature cost.

    Args:
        R (TwoLevelRecourseSet): The recourse set.

    Returns:
        float: The reward value.
    """
    U_3 = C_max * epsilon1 * epsilon2
    return U_3 - featureCost(R)


def reward4(R: TwoLevelRecourseSet) -> float:
    """Calculate the reward for recourse set R based on the feature change.


    Args:
        R (TwoLevelRecourseSet): The TwoLevelRecourseSet for which the reward is calculated.

    Returns:
        float: The calculated reward value.
    """
    U_4 = M_max * epsilon1 * epsilon2
    return U_4 - featureChange(R)
