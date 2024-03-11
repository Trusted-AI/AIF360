import pandas as pd

from aif360.sklearn.detectors.facts.clean import clean_dataset


def test_clean_adult() -> None:
    cols = ["fnlwgt", "education", "relationship", "hours-per-week", "age", "income"]
    mock_adult = pd.DataFrame(
        [
            [13, "hello", " ?", 15, 30, " <=50K"],
            [13, "hello", " waifu", 15, " ?", " <=50K"],
            [13, "hello", " ?", 15, 30, " broke"],
            [" ?", "hello", " Husband", 30, 10, " <=50K"],
            [13, "hello", " Husband", 45, 20, " >50K"],
            [13, "hello", " Husband", 39.5, 30, " <=50K"],
            [13, "hello", " Husband", 15, 40, " >50K"],
            [13, "hello", " Wife", 45, 50, " >50K"],
            [13, "hello", " Wife", 39.5, 60, " <=50K"],
            [13, "hello", " Wife", 80, 70, " >50K"],
        ],
        columns=cols
    )

    cols = ["relationship", "hours-per-week", "age", "income"]
    res_adult = pd.DataFrame(
        [
            ["Married", "MidTime", 10, 0],
            ["Married", "OverTime", 20, 1],
            ["Married", "FullTime", 30, 0],
            ["Married", "PartTime", 40, 1],
            ["Married", "OverTime", 50, 1],
            ["Married", "FullTime", 60, 0],
            ["Married", "BrainDrain", 70, 1],
        ],
        columns=cols
    )
    res_adult["age"] = pd.qcut(res_adult["age"], q=5)

    df1 = clean_dataset(mock_adult, "adult").reset_index(drop=True)
    df2 = res_adult
    assert (df1 == df2).all().all()

def test_clean_ssl() -> None:
    cols = ["SSL SCORE", "RACE CODE CD", "PREDICTOR RAT TREND IN CRIMINAL ACTIVITY", "PREDICTOR RAT AGE AT LATEST ARREST"]
    mock_ssl = pd.DataFrame(
        [
            [200, "WBH", 1, "20-30"],
            [200, "WBH", 2, "20-30"],
            [400, "WBH", 3, "30-40"],
            [200, "WBH", 4, "30-40"],
            [200, "WWH", 5, "20-30"],
            [400, "WWH", 6, "less than 20"],
            [200, "API", 7, "less than 20"],
            [200, "U", 8, "45"],
            [400, "X", 9, "39.5"],
            [200, "I", 10, "80"],
            [200, "API", 11, "80"],
        ],
        columns=cols
    )

    res_ssl = pd.DataFrame(
        [
            [1, "BLK", 1, "20-30"],
            [1, "BLK", 2, "20-30"],
            [0, "BLK", 3, "30-40"],
            [1, "BLK", 4, "30-40"],
            [1, "WHI", 5, "20-30"],
            [0, "WHI", 6, "10-20"],
        ],
        columns=cols
    )
    res_ssl["PREDICTOR RAT TREND IN CRIMINAL ACTIVITY"] = pd.qcut(res_ssl["PREDICTOR RAT TREND IN CRIMINAL ACTIVITY"], q=6)

    df1 = clean_dataset(mock_ssl, "SSL").reset_index(drop=True)
    df2 = res_ssl
    assert (df1 == df2).all().all()

def test_clean_compas() -> None:
    cols = ["age", "c_charge_desc", "priors_count", "age_cat", "target"]
    mock_compas = pd.DataFrame(
        [
            [200, "WBH", 0, "20-30", "Recidivated"],
            [200, "WBH", 1, "20-30", "Recidivated"],
            [400, "WBH", 3, "30-40", "Recidivated"],
            [200, "WBH", 7, "30-40", "Survived"],
            [200, "WWH", 13, "20-30", "Recidivated"],
            [400, "WWH", 14, "Less than 25", "Survived"],
            [200, "API", 17, "Less than 25", "Recidivated"],
            [200, "API", 21, "Less than 25", "Recidivated"],
            [400, "API", 29, "Less than 25", "Survived"],
            [200, "API", 31, "Less than 25", "Recidivated"],
            [200, "API", 37, "Less than 25", "Survived"],
        ],
        columns=cols
    )

    cols = ["priors_count", "age_cat", "target"]
    res_compas = pd.DataFrame(
        [
            [0, "20-30", 0],
            [1, "20-30", 0],
            [3, "30-40", 0],
            [7, "30-40", 1],
            [13, "20-30", 0],
            [14, "10-25", 1],
            [17, "10-25", 0],
            [21, "10-25", 0],
            [29, "10-25", 1],
            [31, "10-25", 0],
            [37, "10-25", 1],
        ],
        columns=cols
    )
    res_compas["priors_count"] = pd.cut(res_compas["priors_count"], [-0.1, 1, 5, 10, 15, 38])

    df1 = clean_dataset(mock_compas, "compas").reset_index(drop=True)
    df2 = res_compas
    assert (df1 == df2).all().all()