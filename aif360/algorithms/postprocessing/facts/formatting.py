from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt

from colorama import Fore, Style

from .predicate import Predicate

ASSUME_ZERO = 10 ** (-7)

def recourse_report_reverse(
    rules: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    population_sizes: Optional[Dict[str, int]] = None,
    missing_subgroup_val: str = "N/A",
    subgroup_costs: Optional[Dict[Predicate, Dict[str, float]]] = None,
    show_subgroup_costs: bool = False,
    show_bias: Optional[str] = None,
) -> str:
    """
    Generates a report detailing the recourses and fairness assessment for a given set of rules.

    Args:
        rules (Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]):
            The collection of rules with associated recourses and correctness.
        population_sizes (Optional[Dict[str, int]], optional):
            A dictionary specifying the population sizes for the subgroups.
            If provided, coverage statistics will be included in the report.
            Defaults to None.
        missing_subgroup_val (str, optional):
            The value to represent missing subgroups in the report.
            Defaults to "N/A".
        subgroup_costs (Optional[Dict[Predicate, Dict[str, float]]], optional):
            A dictionary specifying the aggregate costs for each subgroup in each rule.
            If provided, the costs of recourses will be included in the report.
            Defaults to None.
        show_subgroup_costs (bool, optional):
            Indicates whether to display the subgroup costs in the report.
            Only applicable when subgroup_costs is provided.
            Defaults to False.
        show_bias (Optional[str], optional):
            Specifies the biased subgroup to highlight in the report.
            Only applicable when subgroup_costs is provided.
            Defaults to None.

    Returns:
        str: The generated report as a string.
    """
    if len(rules) == 0:
        return f"{Style.BRIGHT}With the given parameters, no recourses showing unfairness have been found!{Style.RESET_ALL}\n"

    ret = []
    for ifclause, sg_thens in rules.items():
        if subgroup_costs is not None and show_bias is not None:
            curr_subgroup_costs = subgroup_costs[ifclause]
            max_intergroup_cost_diff = max(curr_subgroup_costs.values()) - min(
                curr_subgroup_costs.values()
            )
            biased_subgroup, max_cost = max(
                curr_subgroup_costs.items(), key=lambda p: p[1]
            )
            if biased_subgroup != show_bias:
                continue

        ret.append(f"If {Style.BRIGHT}{ifclause}{Style.RESET_ALL}:\n")
        for subgroup, (cov, thens) in sg_thens.items():
            if subgroup == missing_subgroup_val:
                continue

            # print coverage statistics for the subgroup
            ret.append(
                f"\tProtected Subgroup '{Style.BRIGHT}{subgroup}{Style.RESET_ALL}', {Fore.BLUE}{cov:.2%}{Fore.RESET} covered"
            )
            if population_sizes is not None:
                if subgroup in population_sizes:
                    ret.append(f" out of {population_sizes[subgroup]}")
                else:
                    ret.append(" (protected subgroup population size not given)")
            ret.append("\n")

            # print each available recourse together with the respective correctness
            if thens == []:
                ret.append(
                    f"\t\t{Fore.RED}No recourses for this subgroup!\n{Fore.RESET}"
                )
            for then, correctness in thens:
                _, thenstr = ifthen2str(ifclause=ifclause, thenclause=then)

                # abs() used to get rid of -0.0
                assert correctness >= -ASSUME_ZERO
                cor_str = Fore.GREEN + f"{abs(correctness):.2%}" + Fore.RESET
                ret.append(
                    f"\t\tMake {Style.BRIGHT}{thenstr}{Style.RESET_ALL} with correctness {cor_str}.\n"
                )

            if subgroup_costs is not None and show_subgroup_costs:
                cost_of_current_subgroup = subgroup_costs[ifclause][subgroup]
                if f"{cost_of_current_subgroup:.2f}" == "-0.00":
                    cost_of_current_subgroup = 0
                ret.append(
                    f"\t\t{Style.BRIGHT}Aggregate cost{Style.RESET_ALL} of the above recourses = {Fore.MAGENTA}{cost_of_current_subgroup:.2f}{Fore.RESET}\n"
                )

        # TODO: show bias message in (much) larger font size.
        if subgroup_costs is not None:
            curr_subgroup_costs = subgroup_costs[ifclause]
            max_intergroup_cost_diff = max(curr_subgroup_costs.values()) - min(
                curr_subgroup_costs.values()
            )
            biased_subgroup, max_cost = max(
                curr_subgroup_costs.items(), key=lambda p: p[1]
            )
            if max_intergroup_cost_diff > 0:
                ret.append(
                    f"\t{Fore.MAGENTA}Bias against {biased_subgroup}. Unfairness score = {round(max_intergroup_cost_diff,2)}.{Fore.RESET}\n"
                )
            else:
                ret.append("\tNo bias!\n")

    return "".join(ret)


def print_recourse_report(
    rules: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    population_sizes: Optional[Dict[str, int]] = None,
    missing_subgroup_val: str = "N/A",
    subgroup_costs: Optional[Dict[Predicate, Dict[str, float]]] = None,
    aggregate_cors_costs: Optional[
        Dict[Predicate, Dict[str, List[Tuple[float, float]]]]
    ] = None,
    show_subgroup_costs: bool = False,
    show_bias: Optional[str] = None,
    metric_name: str = "Equal Effectiveness",
) -> None:
    """
    Prints a report detailing the recourses and fairness assessment for a given set of rules.

    Args:
        rules (Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]):
            The collection of rules with associated recourses and correctness.
        population_sizes (Optional[Dict[str, int]], optional):
            A dictionary specifying the population sizes for the subgroups.
            If provided, coverage statistics will be included in the report.
            Defaults to None.
        missing_subgroup_val (str, optional):
            The value to represent missing subgroups in the report.
            Defaults to "N/A".
        subgroup_costs (Optional[Dict[Predicate, Dict[str, float]]], optional):
            A dictionary specifying the aggregate costs for each subgroup in each rule.
            If provided, the costs of recourses will be included in the report.
            Defaults to None.
        aggregate_cors_costs (Optional[Dict[Predicate, Dict[str, List[Tuple[float, float]]]]], optional):
            A dictionary specifying the aggregate correctness and cost for each subgroup in each rule.
            If provided, a cumulative effectiveness plot will be included in the report.
            Defaults to None.
        show_subgroup_costs (bool, optional):
            Indicates whether to display the subgroup costs in the report.
            Only applicable when subgroup_costs is provided.
            Defaults to False.
        show_bias (Optional[str], optional):
            Specifies the biased subgroup to highlight in the report.
            Only applicable when subgroup_costs is provided.
            Defaults to None.
        metric_name (str, optional):
            The name of the fairness metric used to assess bias.
            Only applicable when subgroup_costs is provided.
            Defaults to "Equal Effectiveness".

    Returns:
        None
    """
    if len(rules) == 0:
        print(
            f"{Style.BRIGHT}With the given parameters, no recourses showing unfairness have been found!{Style.RESET_ALL}"
        )

    for ifclause, sg_thens in rules.items():
        if subgroup_costs is not None and show_bias is not None:
            curr_subgroup_costs = subgroup_costs[ifclause]
            max_intergroup_cost_diff = max(curr_subgroup_costs.values()) - min(
                curr_subgroup_costs.values()
            )
            biased_subgroup, max_cost = max(
                curr_subgroup_costs.items(), key=lambda p: p[1]
            )
            if biased_subgroup != show_bias:
                continue

        print(f"If {Style.BRIGHT}{ifclause}{Style.RESET_ALL}:")
        for subgroup, (cov, thens) in sg_thens.items():
            if subgroup == missing_subgroup_val:
                continue

            # print coverage statistics for the subgroup
            print(
                f"\tProtected Subgroup '{Style.BRIGHT}{subgroup}{Style.RESET_ALL}', {Fore.BLUE}{cov:.2%}{Fore.RESET} covered",
                end="",
            )
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
                print(
                    f"\t\tMake {Style.BRIGHT}{thenstr}{Style.RESET_ALL} with effectiveness {cor_str}."
                )

            if subgroup_costs is not None and show_subgroup_costs:
                cost_of_current_subgroup = subgroup_costs[ifclause][subgroup]
                if f"{cost_of_current_subgroup:.2f}" == "-0.00":
                    cost_of_current_subgroup = 0
                print(
                    f"\t\t{Style.BRIGHT}Aggregate cost{Style.RESET_ALL} of the above recourses = {Fore.MAGENTA}{cost_of_current_subgroup:.2f}{Fore.RESET}"
                )

        # TODO: show bias message in (much) larger font size.
        if subgroup_costs is not None:
            curr_subgroup_costs = subgroup_costs[ifclause]
            max_intergroup_cost_diff = max(curr_subgroup_costs.values()) - min(
                curr_subgroup_costs.values()
            )
            biased_subgroup, max_cost = max(
                curr_subgroup_costs.items(), key=lambda p: p[1]
            )
            if max_intergroup_cost_diff > 0:
                print(
                    f"\t{Fore.MAGENTA}Bias against {biased_subgroup} due to {metric_name}. Unfairness score = {round(max_intergroup_cost_diff,2)}.{Fore.RESET}"
                )
            else:
                print(f"\t{Fore.MAGENTA}No bias!{Fore.RESET}")

        if aggregate_cors_costs is not None and ifclause in aggregate_cors_costs:
            print(
                f"\t{Fore.CYAN}Cumulative effectiveness plot for the above recourses:{Fore.RESET}"
            )
            cost_cors = {}
            for sg, thens in aggregate_cors_costs[ifclause].items():
                cost_cors[sg] = ([cost for _, cost in thens], [cor for cor, _ in thens])
            plot_aggregate_correctness(cost_cors)
            plt.show()


def print_recourse_report_cumulative(
    rules: Dict[
        Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]
    ],
    population_sizes: Optional[Dict[str, int]] = None,
    missing_subgroup_val: str = "N/A",
    subgroup_costs: Optional[Dict[Predicate, Dict[str, float]]] = None,
    show_subgroup_costs: bool = False,
    show_then_costs: bool = False,
    show_cumulative_plots: bool = False,
    show_bias: Optional[str] = None,
    correctness_metric: bool = False,
    metric_name: str = "Equal Effectiveness",
) -> None:
    """
    Prints a report detailing the recourses and fairness assessment for a given set of rules.

    Args:
        rules (Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]):
            The collection of rules with associated recourses, correctness, and costs.
        population_sizes (Optional[Dict[str, int]], optional):
            A dictionary specifying the population sizes for the subgroups.
            If provided, coverage statistics will be included in the report.
            Defaults to None.
        missing_subgroup_val (str, optional):
            The value to represent missing subgroups in the report.
            Defaults to "N/A".
        subgroup_costs (Optional[Dict[Predicate, Dict[str, float]]], optional):
            A dictionary specifying the aggregate costs for each subgroup in each rule.
            If provided, the costs of recourses will be included in the report.
            Defaults to None.
        show_subgroup_costs (bool, optional):
            Indicates whether to display the subgroup costs in the report.
            Only applicable when subgroup_costs is provided.
            Defaults to False.
        show_then_costs (bool, optional):
            Indicates whether to display the counterfactual costs of recourses in the report.
            Defaults to False.
        show_cumulative_plots (bool, optional):
            Indicates whether to display the cumulative effectiveness plots in the report.
            Defaults to False.
        show_bias (Optional[str], optional):
            Specifies the biased subgroup to highlight in the report.
            Only applicable when subgroup_costs is provided.
            Defaults to None.
        correctness_metric (bool, optional):
            Indicates whether to use the correctness metric or the cost metric for assessing bias.
            If False, the maximum cost difference will be used.
            If True, the minimum cost difference will be used.
            Only applicable when subgroup_costs is provided.
            Defaults to False.
        metric_name (str, optional):
            The name of the fairness metric used to assess bias.
            Only applicable when subgroup_costs is provided.
            Defaults to "Equal Effectiveness".

    Returns:
        None
    """
    if len(rules) == 0:
        print(
            f"{Style.BRIGHT}With the given parameters, no recourses showing unfairness have been found!{Style.RESET_ALL}"
        )

    for ifclause, sg_thens in rules.items():
        if subgroup_costs is not None and show_bias is not None:
            curr_subgroup_costs = subgroup_costs[ifclause]
            max_intergroup_cost_diff = max(curr_subgroup_costs.values()) - min(
                curr_subgroup_costs.values()
            )
            biased_subgroup, max_cost = max(
                curr_subgroup_costs.items(), key=lambda p: p[1]
            )
            if biased_subgroup != show_bias:
                continue

        print(f"If {Style.BRIGHT}{ifclause}{Style.RESET_ALL}:")
        for subgroup, (cov, thens) in sg_thens.items():
            if subgroup == missing_subgroup_val:
                continue

            # print coverage statistics for the subgroup
            print(
                f"\tProtected Subgroup '{Style.BRIGHT}{subgroup}{Style.RESET_ALL}', {Fore.BLUE}{cov:.2%}{Fore.RESET} covered",
                end="",
            )
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
                print(
                    f"\t\tMake {Style.BRIGHT}{thenstr}{Style.RESET_ALL} with effectiveness {cor_str}",
                    end="",
                )

                if show_then_costs:
                    print(f" and counterfactual cost = {round(cost,2)}", end="")
                print(".")

            if subgroup_costs is not None and show_subgroup_costs:
                cost_of_current_subgroup = subgroup_costs[ifclause][subgroup]
                if f"{cost_of_current_subgroup:.2f}" == "-0.00":
                    cost_of_current_subgroup = 0
                print(
                    f"\t\t{Style.BRIGHT}Aggregate cost{Style.RESET_ALL} of the above recourses = {Fore.MAGENTA}{cost_of_current_subgroup:.2f}{Fore.RESET}"
                )

        # TODO: show bias message in (much) larger font size.
        if subgroup_costs is not None:
            curr_subgroup_costs = subgroup_costs[ifclause]
            if correctness_metric == False:
                max_intergroup_cost_diff = max(curr_subgroup_costs.values()) - min(
                    curr_subgroup_costs.values()
                )
                biased_subgroup, max_cost = max(
                    curr_subgroup_costs.items(), key=lambda p: p[1]
                )
            else:
                max_intergroup_cost_diff = max(curr_subgroup_costs.values()) - min(
                    curr_subgroup_costs.values()
                )
                biased_subgroup, max_cost = min(
                    curr_subgroup_costs.items(), key=lambda p: p[1]
                )
            if max_intergroup_cost_diff > 0:
                print(
                    f"\t{Fore.MAGENTA}Bias against {biased_subgroup} due to {metric_name}. Unfairness score = {round(max_intergroup_cost_diff,3)}.{Fore.RESET}"
                )
            else:
                print(f"\t{Fore.MAGENTA}No bias!{Fore.RESET}")

        if show_cumulative_plots:
            print(
                f"\t{Fore.CYAN}Cumulative effectiveness plot for the above recourses:{Fore.RESET}"
            )
            cost_cors = {}
            for sg, (_cov, thens) in rules[ifclause].items():
                cost_cors[sg] = (
                    [cost for _, _, cost in thens],
                    [cor for _, cor, _ in thens],
                )
            plot_aggregate_correctness(cost_cors)
            plt.show()


def print_recourse_report_KStest_cumulative(
    rules: Dict[
        Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]
    ],
    population_sizes: Optional[Dict[str, int]] = None,
    missing_subgroup_val: str = "N/A",
    unfairness: Optional[Dict[Predicate, float]] = None,
    show_then_costs: bool = False,
    show_cumulative_plots: bool = False,
    metric_name="Fair Effectiveness-Cost Trade-Off",
) -> None:
    """
    Prints a report detailing the recourses and fairness assessment for a given set of rules.

    Args:
        rules (Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]):
            The collection of rules with associated recourses, correctness, and costs.
        population_sizes (Optional[Dict[str, int]], optional):
            A dictionary specifying the population sizes for the subgroups.
            If provided, coverage statistics will be included in the report.
            Defaults to None.
        missing_subgroup_val (str, optional):
            The value to represent missing subgroups in the report.
            Defaults to "N/A".
        unfairness (Optional[Dict[Predicate, float]], optional):
            A dictionary specifying the unfairness scores for each rule.
            If provided, the unfairness scores will be included in the report.
            Defaults to None.
        show_then_costs (bool, optional):
            Indicates whether to display the counterfactual costs of recourses in the report.
            Defaults to False.
        show_cumulative_plots (bool, optional):
            Indicates whether to display the cumulative effectiveness plots in the report.
            Defaults to False.
        metric_name (str, optional):
            The name of the fairness metric used to assess unfairness.
            Only applicable when unfairness is provided.
            Defaults to "Fair Effectiveness-Cost Trade-Off".

    Returns:
        None
    """
    if len(rules) == 0:
        print(
            f"{Style.BRIGHT}With the given parameters, no recourses showing unfairness have been found!{Style.RESET_ALL}"
        )

    for ifclause, sg_thens in rules.items():
        print(f"If {Style.BRIGHT}{ifclause}{Style.RESET_ALL}:")
        for subgroup, (cov, thens) in sg_thens.items():
            if subgroup == missing_subgroup_val:
                continue

            # print coverage statistics for the subgroup
            print(
                f"\tProtected Subgroup '{Style.BRIGHT}{subgroup}{Style.RESET_ALL}', {Fore.BLUE}{cov:.2%}{Fore.RESET} covered",
                end="",
            )
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
                print(
                    f"\t\tMake {Style.BRIGHT}{thenstr}{Style.RESET_ALL} with effectiveness {cor_str}",
                    end="",
                )

                if show_then_costs:
                    print(f" and counterfactual cost = {round(cost,2)}", end="")
                print(".")

        if unfairness is not None:
            curr_subgroup_costs = unfairness[ifclause]
            print(
                f"\t{Fore.MAGENTA} Unfairness based on the {metric_name} = {round(curr_subgroup_costs,2)}.{Fore.RESET}"
            )

        if show_cumulative_plots:
            print(
                f"\t{Fore.CYAN}Cumulative effectiveness plot for the above recourses:{Fore.RESET}"
            )
            cost_cors = {}
            for sg, (_cov, thens) in rules[ifclause].items():
                cost_cors[sg] = (
                    [cost for _, _, cost in thens],
                    [cor for _, cor, _ in thens],
                )
            plot_aggregate_correctness(cost_cors)
            plt.show()


def ifthen2str(
    ifclause: Predicate,
    thenclause: Predicate,
    show_same_feats: bool = False,
    same_col: str = "default",
    different_col: str = Fore.RED,
) -> Tuple[str, str]:
    """
    Converts if-then clauses into strings for display in the recourse report.

    Args:
        ifclause (Predicate): The if-clause predicate.
        thenclause (Predicate): The then-clause predicate.
        show_same_feats (bool, optional):
            Indicates whether to include features with the same values in the string representation.
            Defaults to False.
        same_col (str, optional):
            The color code to use for displaying features with the same values.
            Set to "default" to use the default color.
            Defaults to "default".
        different_col (str, optional):
            The color code to use for displaying features with different values.
            Defaults to Fore.RED.

    Returns:
        Tuple[str, str]: A tuple containing the string representation of the if-clause and then-clause, respectively.
    """
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
    """
    Creates a line plot showing the aggregate correctness percentage for different subgroups based on the cost of change.

    Args:
        costs_cors_per_subgroup (Dict[str, Tuple[List[float], List[float]]]):
            A dictionary mapping subgroup names to a tuple of lists representing the costs and correctness percentages
            for each subgroup.

    Returns:
        plt.Figure: The created matplotlib Figure object.
    """
    subgroup_markers = {
        sg: (index, 0, 0)
        for index, sg in enumerate(costs_cors_per_subgroup.keys(), start=3)
    }
    fig, ax = plt.subplots()
    lines = []
    labels = []
    for sg, (costs, correctnesses) in costs_cors_per_subgroup.items():
        (line,) = ax.step(
            costs,
            correctnesses,
            where="post",
            marker=subgroup_markers[sg],
            label=sg,
            alpha=0.7,
        )
        lines.append(line)
        labels.append(sg)
    ax.set_xlabel("Cost of change")
    ax.set_ylabel("Correctness percentage")
    ax.legend(lines, labels)
    return fig
