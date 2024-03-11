from aif360.sklearn.detectors.facts.formatting import (
    print_recourse_report,
    print_recourse_report_KStest_cumulative,
    ifthen2str,
    plot_aggregate_correctness
)
from aif360.sklearn.detectors.facts.predicate import Predicate

def test_print_recourse_report(capfd) -> None:
    ifthens = {
        Predicate.from_dict({"a": 13}):
        {"Male": (0.2, [
            (Predicate.from_dict({"a": 15}), 0.2, 6.),
            (Predicate.from_dict({"a": 17}), 0.3, 12.),
            (Predicate.from_dict({"a": 19}), 0.5, 18.),
            (Predicate.from_dict({"a": 23}), 0.7, 30.),
        ]),
        "Female": (0.25, [
            (Predicate.from_dict({"a": 15}), 0.15, 6.),
            (Predicate.from_dict({"a": 17}), 0.3, 12.),
            (Predicate.from_dict({"a": 19}), 0.45, 18.),
            (Predicate.from_dict({"a": 23}), 0.65, 30.),
        ])},
        Predicate.from_dict({"a": 13, "b": 45}):
        {"Male": (0.2, [
            (Predicate.from_dict({"a": 15, "b": 40}), 0.45, float(6 + 25)),
            (Predicate.from_dict({"a": 17, "b": 38}), 0.5, float(12 + 35)),
            (Predicate.from_dict({"a": 19, "b": 35}), 0.75, float(18 + 50)),
            (Predicate.from_dict({"a": 23, "b": 33}), 0.99, float(30 + 60)),
        ]),
        "Female": (0.25, [
            (Predicate.from_dict({"a": 15, "b": 40}), 0.4, float(6 + 25)),
            (Predicate.from_dict({"a": 17, "b": 38}), 0.5, float(12 + 35)),
            (Predicate.from_dict({"a": 19, "b": 35}), 0.7, float(18 + 50)),
            (Predicate.from_dict({"a": 23, "b": 33}), 0.85, float(30 + 60)),
        ])},
        Predicate.from_dict({"b": 45}):
        {"Male": (0.2, [
            (Predicate.from_dict({"b": 40}), 0.2, 25.),
            (Predicate.from_dict({"b": 38}), 0.4, 35.),
            (Predicate.from_dict({"b": 35}), 0.6, 50.),
            (Predicate.from_dict({"b": 33}), 0.9, 60.),
        ]),
        "Female": (0.25, [
            (Predicate.from_dict({"b": 40}), 0.15, 25.),
            (Predicate.from_dict({"b": 38}), 0.35, 35.),
            (Predicate.from_dict({"b": 35}), 0.5, 50.),
            (Predicate.from_dict({"b": 33}), 0.8, 60.),
        ])},
    }
    pop_sizes = {"Male": 900, "Female": 1000}
    sg_costs = {
        Predicate.from_dict({"a": 13}): {"Male": 18., "Female": 30.},
        Predicate.from_dict({"a": 13, "b": 45}): {"Male": 47., "Female": 47.},
        Predicate.from_dict({"b": 45}): {"Male": 53., "Female": 88.}
    }

    print_recourse_report(ifthens, pop_sizes, subgroup_costs=sg_costs, show_subgroup_costs=True, show_then_costs=True, show_bias="Male")
    out, err = capfd.readouterr()
    expected = "If \x1b[1ma = 13, b = 45\x1b[0m:\n\tProtected Subgroup '\x1b[1mMale\x1b[0m', \x1b[34m20.00%\x1b[39m covered out of 900\n\t\tMake \x1b[1m\x1b[31ma = 15\x1b[39m, \x1b[31mb = 40\x1b[39m\x1b[0m with effectiveness \x1b[32m45.00%\x1b[39m and counterfactual cost = 31.0.\n\t\tMake \x1b[1m\x1b[31ma = 17\x1b[39m, \x1b[31mb = 38\x1b[39m\x1b[0m with effectiveness \x1b[32m50.00%\x1b[39m and counterfactual cost = 47.0.\n\t\tMake \x1b[1m\x1b[31ma = 19\x1b[39m, \x1b[31mb = 35\x1b[39m\x1b[0m with effectiveness \x1b[32m75.00%\x1b[39m and counterfactual cost = 68.0.\n\t\tMake \x1b[1m\x1b[31ma = 23\x1b[39m, \x1b[31mb = 33\x1b[39m\x1b[0m with effectiveness \x1b[32m99.00%\x1b[39m and counterfactual cost = 90.0.\n\t\t\x1b[1mAggregate cost\x1b[0m of the above recourses = \x1b[35m47.00\x1b[39m\n\tProtected Subgroup '\x1b[1mFemale\x1b[0m', \x1b[34m25.00%\x1b[39m covered out of 1000\n\t\tMake \x1b[1m\x1b[31ma = 15\x1b[39m, \x1b[31mb = 40\x1b[39m\x1b[0m with effectiveness \x1b[32m40.00%\x1b[39m and counterfactual cost = 31.0.\n\t\tMake \x1b[1m\x1b[31ma = 17\x1b[39m, \x1b[31mb = 38\x1b[39m\x1b[0m with effectiveness \x1b[32m50.00%\x1b[39m and counterfactual cost = 47.0.\n\t\tMake \x1b[1m\x1b[31ma = 19\x1b[39m, \x1b[31mb = 35\x1b[39m\x1b[0m with effectiveness \x1b[32m70.00%\x1b[39m and counterfactual cost = 68.0.\n\t\tMake \x1b[1m\x1b[31ma = 23\x1b[39m, \x1b[31mb = 33\x1b[39m\x1b[0m with effectiveness \x1b[32m85.00%\x1b[39m and counterfactual cost = 90.0.\n\t\t\x1b[1mAggregate cost\x1b[0m of the above recourses = \x1b[35m47.00\x1b[39m\n\t\x1b[35mNo bias!\x1b[39m\n"
    assert out == expected

# def test_print_recourse_report_KStest_cumulative() -> None:
#     pass

# def test_ifthen2str() -> None:
#     pass

# def test_plot_aggregate_correctness() -> None:
#     pass

