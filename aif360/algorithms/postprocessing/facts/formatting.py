from typing import List, Dict, Tuple, Any, Optional

import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt

from colorama import Fore, Style

from .models import ModelAPI
from .recourse_sets import TwoLevelRecourseSet
from .metrics import incorrectRecourses, incorrectRecoursesSubmodular, cover, featureChange, featureCost, incorrectRecoursesSingle
from .predicate import Predicate, recIsValid

ASSUME_ZERO = 10**(-7)

def report_base(outer: List[Predicate], blocks: List) -> str:
    ret = []
    for p in outer:
        ret.append(f"If {p}:\n")
        for b in blocks:
            ret.append(f"\t{b}")
    return "".join(ret)

def recourse_report(R: TwoLevelRecourseSet, X_aff: DataFrame, model: ModelAPI) -> str:
    ret = []
    # first, print the statistics of the whole recourse set
    incorrects_additive = incorrectRecourses(R, X_aff, model)
    incorrects_at_least_one = incorrectRecoursesSubmodular(R, X_aff, model)
    coverage = cover(R, X_aff)
    feature_cost = featureCost(R)
    feature_change = featureChange(R)

    ret.append(f"Total coverage: {coverage / X_aff.shape[0]:.3%} (over all affected).\n")
    ret.append(f"Total incorrect recourses: {incorrects_additive / coverage:.3%} (over all those covered).\n")
    if incorrects_at_least_one != incorrects_additive:
        ret.append(f"\tAttention! If measured as at-least-one-correct, it changes to {incorrects_at_least_one / coverage:.3%}!\n")
    ret.append(f"Total feature cost: {feature_cost}.\n")
    ret.append(f"Total feature change: {feature_change}.\n")

    # then, print the rules with the statistics for each rule separately
    sensitive = R.feature
    for val in R.values:
        ret.append(f"If {sensitive} = {val}:\n")
        rules = R.rules[val]
        for h, s in zip(rules.hypotheses, rules.suggestions):
            ret.append(f"\tIf {h},\n\tThen {s}.\n")

            sd = Predicate.from_dict({sensitive: val})
            degenerate_two_level_set = TwoLevelRecourseSet.from_triples([(sd, h, s)])

            coverage = cover(degenerate_two_level_set, X_aff)
            inc_original = incorrectRecoursesSingle(sd, h, s, X_aff, model) / coverage
            inc_submodular = incorrectRecoursesSubmodular(degenerate_two_level_set, X_aff, model) / coverage
            coverage /= X_aff.shape[0]

            ret.append(f"\t\tCoverage: {coverage:.3%} over all affected.\n")
            ret.append(f"\t\tIncorrect recourses additive: {inc_original:.3%} over all individuals covered by this rule.\n")
            ret.append(f"\t\tIncorrect recourses at-least-one: {inc_submodular:.3%} over all individuals covered by this rule.\n")
    return "".join(ret)

def recourse_report_preprocessed(groups: List[str], rules: Dict[str, List[Tuple[Predicate, Predicate, float, float]]]) -> str:
    ret = []
    for val in groups:
        ret.append(f"For subgroup '{val}':\n")
        rules_subgroup = rules[val]
        for h, s, coverage, incorrectness in rules_subgroup:
            ret.append(f"\tIf {h},\n\tThen {s}.\n")

            ret.append(f"\t\tCoverage: {coverage:.3%} of those in the subgroup that are affected.\n")
            ret.append(f"\t\tIncorrect recourses: {incorrectness:.3%} over all individuals covered by this rule.\n")
    return "".join(ret)

def to_bold_str(s: Any) -> str:
    return f"\033[1m{s}\033[0m"

def to_blue_str(s: Any) -> str:
    return f"\033[0;34m{s}\033[0m"

def to_green_str(s: Any) -> str:
    return f"\033[0;32m{s}\033[0m"

def to_red_str(s: Any) -> str:
    return f"\033[0;31m{s}\033[0m"

class ifgroup:
    ifclause: Predicate
    allthens: Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]

def recourse_report_reverse(
    rules: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    population_sizes: Optional[Dict[str, int]] = None,
    missing_subgroup_val: str = "N/A",
    subgroup_costs: Optional[Dict[Predicate, Dict[str, float]]] = None,
    show_subgroup_costs: bool = False,
    show_bias: Optional[str] = None
) -> str:
    if len(rules) == 0:
        return f"{Style.BRIGHT}With the given parameters, no recourses showing unfairness have been found!{Style.RESET_ALL}\n"
    
    ret = []
    for ifclause, sg_thens in rules.items():
        if subgroup_costs is not None and show_bias is not None:
            curr_subgroup_costs = subgroup_costs[ifclause]
            max_intergroup_cost_diff = max(curr_subgroup_costs.values()) - min(curr_subgroup_costs.values())
            biased_subgroup, max_cost = max(curr_subgroup_costs.items(), key=lambda p: p[1])
            if biased_subgroup != show_bias:
                continue
        
        ret.append(f"If {Style.BRIGHT}{ifclause}{Style.RESET_ALL}:\n")
        for subgroup, (cov, thens) in sg_thens.items():
            if subgroup == missing_subgroup_val:
                continue

            # print coverage statistics for the subgroup
            ret.append(f"\tProtected Subgroup '{Style.BRIGHT}{subgroup}{Style.RESET_ALL}', {Fore.BLUE}{cov:.2%}{Fore.RESET} covered")
            if population_sizes is not None:
                if subgroup in population_sizes:
                    ret.append(f" out of {population_sizes[subgroup]}")
                else:
                    ret.append(" (protected subgroup population size not given)")
            ret.append("\n")

            # print each available recourse together with the respective correctness
            if thens == []:
                ret.append(f"\t\t{Fore.RED}No recourses for this subgroup!\n{Fore.RESET}")
            for then, correctness in thens:
                _, thenstr = ifthen2str(ifclause=ifclause, thenclause=then)

                # abs() used to get rid of -0.0
                assert correctness >= -ASSUME_ZERO
                cor_str = Fore.GREEN + f"{abs(correctness):.2%}" + Fore.RESET
                ret.append(f"\t\tMake {Style.BRIGHT}{thenstr}{Style.RESET_ALL} with correctness {cor_str}.\n")

            if subgroup_costs is not None and show_subgroup_costs:
                cost_of_current_subgroup = subgroup_costs[ifclause][subgroup]
                if f"{cost_of_current_subgroup:.2f}" == "-0.00":
                    cost_of_current_subgroup = 0
                ret.append(f"\t\t{Style.BRIGHT}Aggregate cost{Style.RESET_ALL} of the above recourses = {Fore.MAGENTA}{cost_of_current_subgroup:.2f}{Fore.RESET}\n")
        
        # TODO: show bias message in (much) larger font size.
        if subgroup_costs is not None:
            curr_subgroup_costs = subgroup_costs[ifclause]
            max_intergroup_cost_diff = max(curr_subgroup_costs.values()) - min(curr_subgroup_costs.values())
            biased_subgroup, max_cost = max(curr_subgroup_costs.items(), key=lambda p: p[1])
            if max_intergroup_cost_diff > 0:
                ret.append(f"\t{Fore.MAGENTA}Bias against {biased_subgroup}. Unfairness score = {round(max_intergroup_cost_diff,2)}.{Fore.RESET}\n")
            else:
                ret.append("\tNo bias!\n")

    return "".join(ret)

def print_recourse_report(
    rules: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    population_sizes: Optional[Dict[str, int]] = None,
    missing_subgroup_val: str = "N/A",
    subgroup_costs: Optional[Dict[Predicate, Dict[str, float]]] = None,
    aggregate_cors_costs: Optional[Dict[Predicate, Dict[str, List[Tuple[float, float]]]]] = None,
    show_subgroup_costs: bool = False,
    show_bias: Optional[str] = None,
    metric_name: str = 'Equal Effectiveness'
) -> None:
    if len(rules) == 0:
        print(f"{Style.BRIGHT}With the given parameters, no recourses showing unfairness have been found!{Style.RESET_ALL}")
    
    for ifclause, sg_thens in rules.items():
        if subgroup_costs is not None and show_bias is not None:
            curr_subgroup_costs = subgroup_costs[ifclause]
            max_intergroup_cost_diff = max(curr_subgroup_costs.values()) - min(curr_subgroup_costs.values())
            biased_subgroup, max_cost = max(curr_subgroup_costs.items(), key=lambda p: p[1])
            if biased_subgroup != show_bias:
                continue
        
        print(f"If {Style.BRIGHT}{ifclause}{Style.RESET_ALL}:")
        for subgroup, (cov, thens) in sg_thens.items():
            if subgroup == missing_subgroup_val:
                continue

            # print coverage statistics for the subgroup
            print(f"\tProtected Subgroup '{Style.BRIGHT}{subgroup}{Style.RESET_ALL}', {Fore.BLUE}{cov:.2%}{Fore.RESET} covered", end="")
            if population_sizes is not None:
                if subgroup in population_sizes:
                    print(f" out of {population_sizes[subgroup]}", end="")
                else:
                    print(" (protected subgroup population size not given)", end="")
            print()

            # print each available recourse together with the respective correctness
            if thens == []:
                print(f"\t\t{Fore.RED}No recourses for this subgroup!{Fore.RESET}")
            for then, correctness in thens:
                _, thenstr = ifthen2str(ifclause=ifclause, thenclause=then)

                # abs() used to get rid of -0.0
                assert correctness >= -ASSUME_ZERO
                cor_str = Fore.GREEN + f"{abs(correctness):.2%}" + Fore.RESET
                print(f"\t\tMake {Style.BRIGHT}{thenstr}{Style.RESET_ALL} with effectiveness {cor_str}.")

            if subgroup_costs is not None and show_subgroup_costs:
                cost_of_current_subgroup = subgroup_costs[ifclause][subgroup]
                if f"{cost_of_current_subgroup:.2f}" == "-0.00":
                    cost_of_current_subgroup = 0
                print(f"\t\t{Style.BRIGHT}Aggregate cost{Style.RESET_ALL} of the above recourses = {Fore.MAGENTA}{cost_of_current_subgroup:.2f}{Fore.RESET}")
        
        # TODO: show bias message in (much) larger font size.
        if subgroup_costs is not None:
            curr_subgroup_costs = subgroup_costs[ifclause]
            max_intergroup_cost_diff = max(curr_subgroup_costs.values()) - min(curr_subgroup_costs.values())
            biased_subgroup, max_cost = max(curr_subgroup_costs.items(), key=lambda p: p[1])
            if max_intergroup_cost_diff > 0:
                print(f"\t{Fore.MAGENTA}Bias against {biased_subgroup} due to {metric_name}. Unfairness score = {round(max_intergroup_cost_diff,2)}.{Fore.RESET}")
            else:
                print(f"\t{Fore.MAGENTA}No bias!{Fore.RESET}")

        if aggregate_cors_costs is not None and ifclause in aggregate_cors_costs:
            print(f"\t{Fore.CYAN}Cumulative effectiveness plot for the above recourses:{Fore.RESET}")
            cost_cors = {}
            for sg, thens in aggregate_cors_costs[ifclause].items():
                cost_cors[sg] = ([cost for _, cost in thens], [cor for cor, _ in thens])
            plot_aggregate_correctness(cost_cors)
            plt.show()

def print_recourse_report_cumulative(
    rules: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
    population_sizes: Optional[Dict[str, int]] = None,
    missing_subgroup_val: str = "N/A",
    subgroup_costs: Optional[Dict[Predicate, Dict[str, float]]] = None,
    show_subgroup_costs: bool = False,
    show_then_costs: bool = False,
    show_cumulative_plots: bool = False,
    show_bias: Optional[str] = None,
    correctness_metric : bool = False,
    metric_name : str = 'Equal Effectiveness'
    ) -> None:
    if len(rules) == 0:
        print(f"{Style.BRIGHT}With the given parameters, no recourses showing unfairness have been found!{Style.RESET_ALL}")
    
    for ifclause, sg_thens in rules.items():
        if subgroup_costs is not None and show_bias is not None:
            curr_subgroup_costs = subgroup_costs[ifclause]
            max_intergroup_cost_diff = max(curr_subgroup_costs.values()) - min(curr_subgroup_costs.values())
            biased_subgroup, max_cost = max(curr_subgroup_costs.items(), key=lambda p: p[1])
            if biased_subgroup != show_bias:
                continue
        
        print(f"If {Style.BRIGHT}{ifclause}{Style.RESET_ALL}:")
        for subgroup, (cov, thens) in sg_thens.items():
            if subgroup == missing_subgroup_val:
                continue

            # print coverage statistics for the subgroup
            print(f"\tProtected Subgroup '{Style.BRIGHT}{subgroup}{Style.RESET_ALL}', {Fore.BLUE}{cov:.2%}{Fore.RESET} covered", end="")
            if population_sizes is not None:
                if subgroup in population_sizes:
                    print(f" out of {population_sizes[subgroup]}", end="")
                else:
                    print(" (protected subgroup population size not given)", end="")
            print()

            # print each available recourse together with the respective correctness
            if thens == []:
                print(f"\t\t{Fore.RED}No recourses for this subgroup!{Fore.RESET}")
            for then, correctness, cost in thens:
                _, thenstr = ifthen2str(ifclause=ifclause, thenclause=then)

                # abs() used to get rid of -0.0
                assert correctness >= -ASSUME_ZERO
                cor_str = Fore.GREEN + f"{abs(correctness):.2%}" + Fore.RESET
                print(f"\t\tMake {Style.BRIGHT}{thenstr}{Style.RESET_ALL} with effectiveness {cor_str}", end="")

                if show_then_costs:
                    print(f" and counterfactual cost = {round(cost,2)}", end="")
                print(".")

            if subgroup_costs is not None and show_subgroup_costs:
                cost_of_current_subgroup = subgroup_costs[ifclause][subgroup]
                if f"{cost_of_current_subgroup:.2f}" == "-0.00":
                    cost_of_current_subgroup = 0
                print(f"\t\t{Style.BRIGHT}Aggregate cost{Style.RESET_ALL} of the above recourses = {Fore.MAGENTA}{cost_of_current_subgroup:.2f}{Fore.RESET}")
        
        # TODO: show bias message in (much) larger font size.
        if subgroup_costs is not None:
            curr_subgroup_costs = subgroup_costs[ifclause]
            if correctness_metric == False:
                max_intergroup_cost_diff = max(curr_subgroup_costs.values()) - min(curr_subgroup_costs.values())
                biased_subgroup, max_cost = max(curr_subgroup_costs.items(), key=lambda p: p[1])
            else: 
                max_intergroup_cost_diff = max(curr_subgroup_costs.values()) - min(curr_subgroup_costs.values())
                biased_subgroup, max_cost = min(curr_subgroup_costs.items(), key=lambda p: p[1])
            if max_intergroup_cost_diff > 0:
                print(f"\t{Fore.MAGENTA}Bias against {biased_subgroup} due to {metric_name}. Unfairness score = {round(max_intergroup_cost_diff,3)}.{Fore.RESET}")
            else:
                print(f"\t{Fore.MAGENTA}No bias!{Fore.RESET}")

        if show_cumulative_plots:
            print(f"\t{Fore.CYAN}Cumulative effectiveness plot for the above recourses:{Fore.RESET}")
            cost_cors = {}
            for sg, (_cov, thens) in rules[ifclause].items():
                cost_cors[sg] = ([cost for _, _, cost in thens], [cor for _, cor, _ in thens])
            plot_aggregate_correctness(cost_cors)
            plt.show()

def print_recourse_report_KStest_cumulative(
    rules: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
    population_sizes: Optional[Dict[str, int]] = None,
    missing_subgroup_val: str = "N/A",
    unfairness: Optional[Dict[Predicate, float]] = None,
    show_then_costs: bool = False,
    show_cumulative_plots: bool = False,
    metric_name = 'Fair Effectiveness-Cost Trade-Off'
) -> None:
    if len(rules) == 0:
        print(f"{Style.BRIGHT}With the given parameters, no recourses showing unfairness have been found!{Style.RESET_ALL}")
    
    for ifclause, sg_thens in rules.items():
        print(f"If {Style.BRIGHT}{ifclause}{Style.RESET_ALL}:")
        for subgroup, (cov, thens) in sg_thens.items():
            if subgroup == missing_subgroup_val:
                continue

            # print coverage statistics for the subgroup
            print(f"\tProtected Subgroup '{Style.BRIGHT}{subgroup}{Style.RESET_ALL}', {Fore.BLUE}{cov:.2%}{Fore.RESET} covered", end="")
            if population_sizes is not None:
                if subgroup in population_sizes:
                    print(f" out of {population_sizes[subgroup]}", end="")
                else:
                    print(" (protected subgroup population size not given)", end="")
            print()

            # print each available recourse together with the respective correctness
            if thens == []:
                print(f"\t\t{Fore.RED}No recourses for this subgroup!{Fore.RESET}")
            for then, correctness, cost in thens:
                _, thenstr = ifthen2str(ifclause=ifclause, thenclause=then)

                # abs() used to get rid of -0.0
                assert correctness >= -ASSUME_ZERO
                cor_str = Fore.GREEN + f"{abs(correctness):.2%}" + Fore.RESET
                print(f"\t\tMake {Style.BRIGHT}{thenstr}{Style.RESET_ALL} with effectiveness {cor_str}", end="")

                if show_then_costs:
                    print(f" and counterfactual cost = {round(cost,2)}", end="")
                print(".")

            
        if unfairness is not None:
                curr_subgroup_costs = unfairness[ifclause]
                print(f"\t{Fore.MAGENTA} Unfairness based on the {metric_name} = {round(curr_subgroup_costs,2)}.{Fore.RESET}")
    

        if show_cumulative_plots:
            print(f"\t{Fore.CYAN}Cumulative effectiveness plot for the above recourses:{Fore.RESET}")
            cost_cors = {}
            for sg, (_cov, thens) in rules[ifclause].items():
                cost_cors[sg] = ([cost for _, _, cost in thens], [cor for _, cor, _ in thens])
            plot_aggregate_correctness(cost_cors)
            plt.show()

def ifthen2str(
    ifclause: Predicate,
    thenclause: Predicate,
    show_same_feats: bool = False,
    same_col: str = "default",
    different_col: str = Fore.RED
) -> Tuple[str, str]:
    # if not recIsValid(ifclause, thenclause,drop_infeasible):
    #     raise ValueError("If and then clauses should be compatible.")
    
    ifstr = []
    thenstr = []
    first_rep = True
    thendict = thenclause.to_dict()
    for f, v in ifclause.to_dict().items():
        if not show_same_feats and v == thendict[f]:
            continue

        if first_rep:
            first_rep = False
        else:
            ifstr.append(", ")
            thenstr.append(", ")
        
        if v == thendict[f]:
            if same_col != "default":
                ifstr.append(same_col + f"{f} = {v}" + Fore.RESET)
                thenstr.append(same_col + f"{f} = {v}" + Fore.RESET)
            else:
                ifstr.append(f"{f} = {v}")
                thenstr.append(f"{f} = {v}")
        else:
            ifstr.append(different_col + f"{f} = {v}" + Fore.RESET)
            thenstr.append(different_col + f"{f} = {thendict[f]}" + Fore.RESET)
    
    return "".join(ifstr), "".join(thenstr)




def plot_aggregate_correctness(
    costs_cors_per_subgroup: Dict[str, Tuple[List[float], List[float]]]
):
    subgroup_markers = {sg: (index, 0, 0) for index, sg in enumerate(costs_cors_per_subgroup.keys(), start=3)}
    fig, ax = plt.subplots()
    lines = []
    labels = []
    for sg, (costs, correctnesses) in costs_cors_per_subgroup.items():
        line, = ax.step(
            costs,
            correctnesses,
            where="post",
            marker=subgroup_markers[sg],
            label=sg,
            alpha=0.7
        )
        lines.append(line)
        labels.append(sg)
    ax.set_xlabel("Cost of change")
    ax.set_ylabel("Correctness percentage")
    ax.legend(lines, labels)
    return fig
