from typing import Dict, List, Tuple, Any
import pickle
from pathlib import Path
from os import PathLike
from pandas import DataFrame

from .predicate import Predicate


def load_object(file: PathLike) -> object:
    """Loads and returns an object from the specified file using the pickle
        library.

    Args:
        file (PathLike): The path to the file containing the object.

    Returns:
        object: The loaded object.

    Raises:
        None
    """
    p = Path(file)
    with p.open("rb") as inf:
        ret = pickle.load(inf)
    return ret


def save_object(file: PathLike, o: object) -> None:
    """Saves the provided object to the specified file using the pickle library.

    Args:
        file (PathLike): The path to the file where the object will be saved.
        o (object): The object to be saved.

    Returns:
        None

    Raises:
        None
    """
    p = Path(file)
    with p.open("wb") as outf:
        pickle.dump(o, outf)


def load_rules_by_if(
    file: PathLike,
) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]:
    """Loads and returns a dictionary of rules.

    Args:
        file (PathLike): The path to the file containing the rules.

    Returns:
        Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]:
            The dictionary of rules organized by the antecedent Predicate.

    Raises:
        None
    """
    p = Path(file)
    with p.open("rb") as inf:
        rules_by_if = pickle.load(inf)
    return rules_by_if


def save_rules_by_if(
    file: PathLike,
    rules: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
) -> None:
    """Saves the provided rules dictionary to the specified file using the
        pickle library.

    Args:
        file (PathLike): The path to the file where the rules will be saved.
        rules (Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]):
            The dictionary of rules

    Raises:
        None
    """
    p = Path(file)
    with p.open("wb") as outf:
        pickle.dump(rules, outf)


def load_test_data_used(file: PathLike) -> DataFrame:
    """Loads and returns the test data used from the specified file using the
        pickle library.

    Args:
        file (PathLike): The path to the file containing the test data.

    Returns:
        DataFrame: The loaded test data.

    Raises:
        None
    """
    p = Path(file)
    with p.open("rb") as inf:
        X_test = pickle.load(inf)
    return X_test


def save_test_data_used(file: PathLike, X: DataFrame) -> None:
    """Saves the provided test data to the specified file using the pickle
        library.

    Args:
        file (PathLike): The path to the file where the test data will
            be saved.
        X (DataFrame): The test data to be saved.

    Raises:
        None
    """
    p = Path(file)
    with p.open("wb") as outf:
        pickle.dump(X, outf)


def load_model(file: PathLike):
    """Loads and returns a trained model from the specified file using
        the pickle library.

    Args:
        file (PathLike): The path to the file containing the model.

    Returns:
        ModelAPI: The loaded trained model.

    Raises:
        None
    """
    p = Path(file)
    with p.open("rb") as inf:
        model = pickle.load(inf)
    return model


def save_model(file: PathLike, model) -> None:
    """Saves the provided model to the specified file using the pickle
        library.

    Args:
        file (PathLike): The path to the file where the model will be saved.
        model (ModelAPI): The model to be saved.

    Raises:
        None
    """
    p = Path(file)
    with p.open("wb") as outf:
        pickle.dump(model, outf)


def load_state(file: PathLike) -> Tuple[Dict, DataFrame, Any]:
    """Loads and returns the rules, Dataframe, and a model from the specified
        file using the pickle library.

    Args:
        file (PathLike):  The path to the file containing the state.

    Returns:
        Tuple[Dict, DataFrame, ModelAPI]: A tuple containing the loaded rules,
            DataFrame, and model.

    Raises:
        Nones
    """
    p = Path(file)
    with p.open("rb") as inf:
        (rules, X, model) = pickle.load(inf)
    return (rules, X, model)


def save_state(file: PathLike, rules: Dict, X: DataFrame, model) -> None:
    """Saves the rules, dataframe, model to the specified file using the pickle
        library.

    Args:
        file (PathLike): The path to the file where the data will be saved.
        rules (Dict): The rules dictionary to be saved.
        X (DataFrame): The DataFrame to be saved.
        model (ModelAPI): The model to be saved.

    Raises:
        None
    """
    p = Path(file)
    with p.open("wb") as outf:
        pickle.dump((rules, X, model), outf)

