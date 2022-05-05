from slappsim.PetriApp import PetriApp
from slappsim.Function import Function
from slappsim.Structures import Structure
from functools import partial
import numpy as np
from typing import Union


class AppTuner:
    """Serverless Application Tuner

    Serverless application tuner for optimizing the performance and cost of serverless applications.

    Attributes:
        petri_model: A CSPN model of the serverless application.
        performance_profile: A dictionary of the performance profile of functions. The key is the serverless function name, and the value is a dictionary that contains the arrays of the response time (W) under different viable memory options.
        cpu_count: An integer of the number of CPU cores (for parallel computing).
    """

    def __init__(self, petri_model: PetriApp, performance_profile: dict, cpu_count: int = None):
        self.petri_model = petri_model
        self.cpu_count = cpu_count
        self.performance_profile = performance_profile
        # functions for generating probabilistic firing delay under different memory sizes
        self.performance_profile_function_map = {}
        self.function_names = list(self.performance_profile.keys())
        self.performance_profile_avg_rt_map = {}
        self.performance_profile_avg_cost_map = {}
        self.performance_profile_percentile_rt_map = {}
        self.performance_profile_percentile_cost_map = {}
        self.performance_profile_avg_rt_cost_ratio_map_based_on_minimum_cost = {}
        self.performance_profile_avg_rt_cost_ratio_map_based_on_maximum_rt = {}
        self.minimum_cost_configuration = {}
        self.best_performance_configuration = {}
        self.maximum_cost_configuration = {}
        self.worst_performance_configuration = {}
        self.initialize_performance_profile()
        self.get_minimum_cost_configuration()
        self.get_best_performance_configuration()
        self.get_maximum_cost_configuration()
        self.get_worst_performance_configuration()
        self.profile_logs = []
        self.profile_lft_map = {transition.uid: [] for transition in petri_model.transitions}
        self.profile_required_time_map = {transition.uid: [] for transition in petri_model.transitions}
        self.profile_incurred_cost_map = {transition.uid: [] for transition in petri_model.transitions}
        self.ert_list = []
        self.cost_list = []
        self.initialize_with_minimum_cost_configuration()
        self.profile_multiprocessing(k=10000)
        self.avg_ert_under_minimum_cost_configuration = np.mean(self.ert_list)
        self.avg_cost_under_minimum_cost_configuration = np.mean(self.cost_list)
        self.ert_list_under_minimum_cost_configuration = self.ert_list
        self.cost_list_under_minimum_cost_configuration = self.cost_list
        self.percentile_95th_ert_under_minimum_cost_configuration = np.percentile(a=self.ert_list, q=95)
        self.percentile_95th_cost_under_minimum_cost_configuration = np.percentile(a=self.cost_list, q=95)
        self.initialize_with_best_performance_configuration()
        self.profile_multiprocessing(k=10000)
        self.avg_ert_under_best_performance_configuration = np.mean(self.ert_list)
        self.avg_cost_under_best_performance_configuration = np.mean(self.cost_list)
        self.ert_list_under_best_performance_configuration = self.ert_list
        self.cost_list_under_best_performance_configuration = self.cost_list
        self.percentile_95th_ert_under_best_performance_configuration = np.percentile(a=self.ert_list, q=95)
        self.percentile_95th_cost_under_best_performance_configuration = np.percentile(a=self.cost_list, q=95)
        self.early_reject_threshold = 1.1

    def initialize_performance_profile(self) -> None:
        """Initialize performance_profile_function_map, performance_profile_avg_rt_map,
        performance_profile_avg_cost_map. """
        dic = {}
        avg_rt_dic = {}
        avg_cost_dic = {}
        avg_rt_cost_ratio_dict_based_on_minimum_cost = {}
        avg_rt_cost_ratio_dict_based_on_maximum_rt = {}
        for function in self.performance_profile.keys():
            profile = {}
            avg_rt = {}
            avg_cost = {}
            avg_rt_cost_ratio_based_on_minimum_cost = {}
            avg_rt_cost_ratio_based_on_maximum_rt = {}
            for memory in self.performance_profile[function].keys():
                self.performance_profile[function][memory] = np.array(self.performance_profile[function][memory])
                pf_fun = partial(np.random.choice, a=self.performance_profile[function][memory])
                profile[memory] = pf_fun
                avg_rt[memory] = np.mean(self.performance_profile[function][memory])
                function_transition = [fun for fun in self.petri_model.functions if fun.name == function][0]
                cost_fun = partial(function_transition.calculate_cost, mem=memory, pmms=self.petri_model.pmms,
                                   ppi=self.petri_model.ppi)
                cost_fun_vec = np.vectorize(cost_fun)
                avg_cost[memory] = np.round(np.mean(cost_fun_vec(self.performance_profile[function][memory]) * 1000000),
                                            7)
            maximum_rt_mem = max(avg_rt, key=avg_rt.get)
            maximum_rt = avg_rt[maximum_rt_mem]
            maximum_rt_cost = avg_cost[maximum_rt_mem]
            for memory in self.performance_profile[function].keys():
                if memory != maximum_rt_mem:
                    avg_rt_cost_ratio_based_on_maximum_rt[memory] = (maximum_rt - avg_rt[memory]) / (
                            maximum_rt_cost - avg_cost[memory])
                else:
                    avg_rt_cost_ratio_based_on_maximum_rt[memory] = 0
            dic[function] = profile
            avg_rt_dic[function] = avg_rt
            avg_cost_dic[function] = avg_cost
            avg_rt_cost_ratio_dict_based_on_minimum_cost[function] = avg_rt_cost_ratio_based_on_minimum_cost
            avg_rt_cost_ratio_dict_based_on_maximum_rt[function] = avg_rt_cost_ratio_based_on_maximum_rt
        self.performance_profile_function_map = dic
        self.performance_profile_avg_rt_map = avg_rt_dic
        self.performance_profile_avg_cost_map = avg_cost_dic
        self.performance_profile_avg_rt_cost_ratio_map_based_on_minimum_cost = avg_rt_cost_ratio_dict_based_on_minimum_cost
        self.performance_profile_avg_rt_cost_ratio_map_based_on_maximum_rt = avg_rt_cost_ratio_dict_based_on_maximum_rt

    def update_performance_profile(self, function_name: str, memory: int) -> None:
        """Update the memory size and the performance profile of the serverless function."""
        if self.performance_profile.get(function_name, {}).get(memory, None) is None:
            return
        if self.petri_model.functions_map[function_name] in self.petri_model.transitions:
            self.petri_model.functions_map[function_name].pf_fun = self.performance_profile_function_map[function_name][
                memory]
            self.petri_model.functions_map[function_name].mem = memory
        else:
            # If the function is in a map structure.
            for function in self.petri_model.transitions:
                if isinstance(function, Function) and function.copied_from == \
                        self.petri_model.functions_map[function_name]:
                    function.pf_fun = self.performance_profile_function_map[function_name][memory]
                    function.mem = memory
                    self.petri_model.functions_map[function_name].mem = memory

    def get_minimum_cost_configuration(self) -> dict:
        """Get the configuration that leads to the minimum cost."""
        dic = {}
        for function in self.petri_model.functions:
            dic[function.name] = min(self.performance_profile_avg_cost_map[function.name],
                                     key=self.performance_profile_avg_cost_map[function.name].get)
        self.minimum_cost_configuration = dic
        return self.minimum_cost_configuration

    def get_best_performance_configuration(self) -> dict:
        """Get the configuration that leads to the best performance (the shortest end-to-end response time)."""
        dic = {}
        for function in self.petri_model.functions:
            dic[function.name] = min(self.performance_profile_avg_rt_map[function.name],
                                     key=self.performance_profile_avg_rt_map[function.name].get)
        self.best_performance_configuration = dic
        return self.best_performance_configuration

    def get_maximum_cost_configuration(self) -> dict:
        """Get the configuration that leads to the maximum cost."""
        dic = {}
        for function in self.petri_model.functions:
            dic[function.name] = max(self.performance_profile_avg_cost_map[function.name],
                                     key=self.performance_profile_avg_cost_map[function.name].get)
        self.maximum_cost_configuration = dic
        return self.maximum_cost_configuration

    def get_worst_performance_configuration(self) -> dict:
        """Get the configuration that leads to the worst performance."""
        dic = {}
        for function in self.petri_model.functions:
            dic[function.name] = max(self.performance_profile_avg_rt_map[function.name],
                                     key=self.performance_profile_avg_rt_map[function.name].get)
        self.worst_performance_configuration = dic
        return self.worst_performance_configuration

    def initialize_with_minimum_cost_configuration(self) -> None:
        """Apply the configuration that leads to the minimum cost."""
        for function in self.minimum_cost_configuration.keys():
            self.update_performance_profile(function, self.minimum_cost_configuration[function])

    def initialize_with_best_performance_configuration(self) -> None:
        """Apply the configuration that leads to the best performance."""
        for function in self.best_performance_configuration.keys():
            self.update_performance_profile(function, self.best_performance_configuration[function])

    def initialize_with_worst_performance_configuration(self) -> None:
        """Apply the configuration that leads to the worst performance."""
        for function in self.worst_performance_configuration.keys():
            self.update_performance_profile(function, self.worst_performance_configuration[function])

    def initialize_with_maximum_cost_configuration(self) -> None:
        """Apply the configuration that leads to the maximum cost."""
        for function in self.maximum_cost_configuration.keys():
            self.update_performance_profile(function, self.maximum_cost_configuration[function])

    def profile(self, k: int) -> None:
        """Obtain the performance, cost, and firing logs of the application under a given configuration."""
        ert_list = np.empty(shape=k)
        cost_list = np.empty(shape=k)
        status_list = []
        log_list = []
        self.profile_lft_map = {transition.uid: [] for transition in self.petri_model.transitions}
        self.profile_required_time_map = {transition.uid: [] for transition in self.petri_model.transitions}
        self.profile_incurred_cost_map = {transition.uid: [] for transition in self.petri_model.transitions}
        for i in range(k):
            rt, c, s, log = self.petri_model.execute()
            ert_list[i] = rt
            cost_list[i] = c
            status_list.append(s)
            log_list.append(log)
            self.petri_model.reset()
        self.profile_logs = log_list
        self.ert_list = ert_list
        self.cost_list = cost_list
        self.process_profiling_logs()

    def profile_multiprocessing(self, k: int) -> None:
        """Obtain the performance, cost, and firing logs of the application under a given configuration. (Parallel
        computing) """
        self.profile_lft_map = {transition.uid: [] for transition in self.petri_model.transitions}
        self.profile_required_time_map = {transition.uid: [] for transition in self.petri_model.transitions}
        self.profile_incurred_cost_map = {transition.uid: [] for transition in self.petri_model.transitions}
        terminal_time, cost, exit_status, firing_logs = self.petri_model.profile(k, self.cpu_count)
        self.profile_logs = firing_logs
        self.ert_list = np.array(terminal_time)
        self.cost_list = np.array(cost)
        self.process_profiling_logs()

    def process_profiling_logs(self) -> None:
        """Preprocess the firing logs."""
        for log in self.profile_logs:
            for transition_uid, lft, required_time, incurred_cost in log:
                self.profile_lft_map[transition_uid].append(lft)
                self.profile_required_time_map[transition_uid].append(required_time)
                self.profile_incurred_cost_map[transition_uid].append(incurred_cost)
        for transition_uid in self.profile_lft_map.keys():
            self.profile_lft_map[transition_uid] = np.array(self.profile_lft_map[transition_uid])
        for transition_uid in self.profile_required_time_map.keys():
            self.profile_required_time_map[transition_uid] = np.array(self.profile_required_time_map[transition_uid])
        for transition_uid in self.profile_incurred_cost_map.keys():
            self.profile_incurred_cost_map[transition_uid] = np.array(self.profile_incurred_cost_map[transition_uid])

    def get_most_significant_impact_function(self, parent_structure=None, check_visited=False, optimized_blocks=None) -> \
            Union[Function, None]:
        """Identify the bottleneck function with the most impact on the overall performance."""
        msi_block = None
        max_total_rt = -1
        blocks_to_check = [block for block in self.petri_model.transitions + self.petri_model.structures if
                           block.parent_structure == parent_structure and (
                                   isinstance(block, Structure) or isinstance(block, Function))]
        if check_visited:
            is_optimized = all([block in optimized_blocks for block in blocks_to_check])
            if is_optimized:
                optimized_blocks.append(parent_structure)
                if parent_structure.copied_from is not None:
                    for structure in self.petri_model.structures:
                        if structure.copied_from == parent_structure.copied_from:
                            optimized_blocks.append(structure)
                return self.get_most_significant_impact_function(parent_structure=msi_block,
                                                                 check_visited=check_visited,
                                                                 optimized_blocks=optimized_blocks)
        for block in blocks_to_check:
            total_rt = -1
            if block.label == 'FunctionExecution' or block.label == 'TaskExecution':
                if check_visited and block in optimized_blocks:
                    continue
                total_rt = np.sum(self.profile_required_time_map[block.uid])
            elif isinstance(block, Structure):
                if block.end_transition is not None:
                    if check_visited and block in optimized_blocks:
                        continue
                    total_rt = np.sum(
                        self.profile_lft_map[block.end_transition.uid] - self.profile_lft_map[
                            block.start_transition.uid])
                else:
                    continue
            if total_rt > max_total_rt:
                max_total_rt = total_rt
                msi_block = block
        if msi_block is None:
            return None
        if isinstance(msi_block, Structure):
            return self.get_most_significant_impact_function(parent_structure=msi_block, check_visited=check_visited,
                                                             optimized_blocks=optimized_blocks)
        elif msi_block.label == 'FunctionExecution' or msi_block.label == 'TaskExecution':
            return msi_block

    def get_total_cost(self, block) -> float:
        """Calculate the cost for a given block(structure) based on the firing logs."""
        total_cost = 0
        if block.label == 'FunctionExecution' or block.label == 'TaskExecution':
            return np.round(np.sum(self.profile_incurred_cost_map[block.uid]), 7)
        else:
            blocks_to_check = [item for item in self.petri_model.transitions + self.petri_model.structures if
                               item.parent_structure == block]
            for item in blocks_to_check:
                total_cost += self.get_total_cost(item)
            return total_cost

    def get_most_significant_cost_impact_function(self, parent_structure=None, check_visited=False,
                                                  optimized_blocks=None) -> \
            Union[Function, None]:
        """Identify the bottleneck function with the most impact on the overall cost."""
        msi_block = None
        max_total_cost = -1
        blocks_to_check = [block for block in self.petri_model.transitions + self.petri_model.structures if
                           block.parent_structure == parent_structure and (
                                   isinstance(block, Structure) or isinstance(block, Function))]
        if check_visited:
            is_optimized = all([block in optimized_blocks for block in blocks_to_check])
            if is_optimized:
                optimized_blocks.append(parent_structure)
                if parent_structure.copied_from is not None:
                    for structure in self.petri_model.structures:
                        if structure.copied_from == parent_structure.copied_from:
                            optimized_blocks.append(structure)
                return self.get_most_significant_cost_impact_function(parent_structure=msi_block,
                                                                      check_visited=check_visited,
                                                                      optimized_blocks=optimized_blocks)
        for block in blocks_to_check:
            total_cost = -1
            if block.label == 'FunctionExecution' or block.label == 'TaskExecution':
                if check_visited and block in optimized_blocks:
                    continue
                total_cost = np.round(np.sum(self.profile_incurred_cost_map[block.uid]), 7)
            elif isinstance(block, Structure):
                if block.end_transition is not None:
                    if check_visited and block in optimized_blocks:
                        continue
                    total_cost = self.get_total_cost(block)
                else:
                    continue
            if total_cost > max_total_cost:
                max_total_cost = total_cost
                msi_block = block
        if msi_block is None:
            return None
        if isinstance(msi_block, Structure):
            return self.get_most_significant_cost_impact_function(parent_structure=msi_block,
                                                                  check_visited=check_visited,
                                                                  optimized_blocks=optimized_blocks)
        elif msi_block.label == 'FunctionExecution' or msi_block.label == 'TaskExecution':
            return msi_block

    def optimize_bpbc(self, budget_constraint: float, percentile: int = None) -> dict:
        """
        Solve the BPBC problem.

        Args:
          budget_constraint: A floating number of the budget constraint (cost per 1 million executions in US dollars).
          percentile: K-th percentile.

        Returns: A dictionary of the optimal memory configurations for serverless functions.

        """
        self.initialize_with_minimum_cost_configuration()
        optimized_blocks = []
        functions_to_be_revisited = []
        ineligible_mem_options_map = {key: [] for key in self.function_names}
        early_reject_map = {key: float('inf') for key in self.function_names}
        skip_dfs_flag = False
        next_msi_function = None
        while True:
            if not skip_dfs_flag:
                msi_function = self.get_most_significant_cost_impact_function(parent_structure=None, check_visited=True,
                                                                              optimized_blocks=optimized_blocks)
                skip_dfs_flag = False
            else:
                msi_function = next_msi_function
            if msi_function is None:
                if functions_to_be_revisited is None:
                    configurations = {fun.name: fun.mem for fun in self.petri_model.functions}
                    return configurations
                else:
                    break
            previous_mem = msi_function.mem
            eligible_mem_options = {key: self.performance_profile_avg_rt_map[msi_function.name][key] for key in
                                    self.performance_profile_avg_rt_map[msi_function.name].keys() if
                                    key not in ineligible_mem_options_map[msi_function.name] and
                                    self.performance_profile_avg_rt_map[msi_function.name][key] <
                                    self.performance_profile_avg_rt_map[msi_function.name][previous_mem]
                                    }
            sorted_mem_options = sorted(eligible_mem_options, key=eligible_mem_options.get)
            if len(sorted_mem_options) == 0:
                optimized_blocks.append(msi_function)
                if msi_function.copied_from is not None:
                    for function in self.petri_model.transitions:
                        if isinstance(function, Function) and function.copied_from == msi_function.copied_from:
                            optimized_blocks.append(function)
                skip_dfs_flag = False
                continue
            constraint_previously_satisfied_flag = False
            ms_flag = False
            minimum_non_msi_cost = float('inf')
            minimum_non_msi_mem = previous_mem
            for i in range(len(sorted_mem_options)):
                mem = sorted_mem_options[i]
                if self.performance_profile_avg_cost_map[msi_function.name][mem] > early_reject_map[
                    msi_function.name] * self.early_reject_threshold:
                    continue
                self.update_performance_profile(msi_function.name, mem)
                self.profile_multiprocessing(k=10000)
                if percentile is None:
                    new_cost = np.mean(self.cost_list)
                else:
                    new_cost = np.percentile(a=self.cost_list, q=percentile)
                if new_cost * 1000000 > budget_constraint:
                    ineligible_mem_options_map[msi_function.name].append(mem)
                    if self.performance_profile_avg_cost_map[msi_function.name][mem] < early_reject_map[
                        msi_function.name]:
                        early_reject_map[msi_function.name] = self.performance_profile_avg_cost_map[msi_function.name][
                            mem]
                    continue
                new_msi_function = self.get_most_significant_impact_function(parent_structure=None, check_visited=True,
                                                                             optimized_blocks=optimized_blocks)
                if new_msi_function == msi_function or (
                        new_msi_function.copied_from is not None
                        and new_msi_function.copied_from == msi_function.copied_from):
                    if not constraint_previously_satisfied_flag:
                        minimum_non_msi_mem = mem
                        optimized_blocks.append(msi_function)
                        if msi_function.copied_from is not None:
                            for function in self.petri_model.transitions:
                                if isinstance(function, Function) and function.copied_from == msi_function.copied_from:
                                    optimized_blocks.append(function)
                        if self.minimum_cost_configuration[msi_function.name] == previous_mem:
                            rt_cost_ratio = {m: self.performance_profile_avg_rt_cost_ratio_map_based_on_maximum_rt[
                                msi_function.name][m] for m in eligible_mem_options.keys() if
                                             self.performance_profile_avg_rt_cost_ratio_map_based_on_maximum_rt[
                                                 msi_function.name][m] > 0 and
                                             self.performance_profile_avg_cost_map[msi_function.name][m] <
                                             self.performance_profile_avg_cost_map[msi_function.name][
                                                 minimum_non_msi_mem] and
                                             self.performance_profile_avg_rt_cost_ratio_map_based_on_maximum_rt[
                                                 msi_function.name][m] >
                                             self.performance_profile_avg_rt_cost_ratio_map_based_on_maximum_rt[
                                                 msi_function.name][minimum_non_msi_mem]
                                             }
                            sorted_optimal_mem_options = sorted(rt_cost_ratio, key=rt_cost_ratio.get,
                                                                reverse=True)
                            rt_cost_ratio = {m: self.performance_profile_avg_rt_cost_ratio_map_based_on_maximum_rt[
                                msi_function.name][m] for m in eligible_mem_options.keys() if
                                             self.performance_profile_avg_rt_cost_ratio_map_based_on_maximum_rt[
                                                 msi_function.name][m] < 0 and
                                             self.performance_profile_avg_cost_map[msi_function.name][m] <
                                             self.performance_profile_avg_cost_map[msi_function.name][
                                                 minimum_non_msi_mem] and
                                             self.performance_profile_avg_rt_cost_ratio_map_based_on_maximum_rt[
                                                 msi_function.name][m] <
                                             self.performance_profile_avg_rt_cost_ratio_map_based_on_maximum_rt[
                                                 msi_function.name][minimum_non_msi_mem]
                                             }
                            sorted_optimal_mem_options += sorted(rt_cost_ratio, key=rt_cost_ratio.get,
                                                                 reverse=False)
                            if len(sorted_optimal_mem_options) == 0:
                                break
                            functions_to_be_revisited.append(msi_function)
                            skip_dfs_flag = False
                            for ms in sorted_optimal_mem_options:
                                if self.performance_profile_avg_cost_map[msi_function.name][ms] > early_reject_map[
                                    msi_function.name] * self.early_reject_threshold:
                                    continue
                                self.update_performance_profile(msi_function.name, ms)
                                self.profile_multiprocessing(k=10000)
                                if percentile is None:
                                    new_cost = np.mean(self.cost_list)
                                else:
                                    new_cost = np.percentile(a=self.cost_list, q=percentile)
                                if new_cost * 1000000 > budget_constraint:
                                    ineligible_mem_options_map[msi_function.name].append(ms)
                                    if self.performance_profile_avg_cost_map[msi_function.name][mem] < early_reject_map[
                                        msi_function.name]:
                                        early_reject_map[msi_function.name] = \
                                            self.performance_profile_avg_cost_map[msi_function.name][mem]
                                    continue
                                else:
                                    ms_flag = True
                                    break
                            if not ms_flag:
                                self.update_performance_profile(msi_function.name, mem)
                                self.profile_multiprocessing(k=10000)
                        break
                    else:
                        break
                else:
                    constraint_previously_satisfied_flag = True
                    if new_cost < minimum_non_msi_cost:
                        minimum_non_msi_cost = new_cost
                        minimum_non_msi_mem = mem
                        next_msi_function = new_msi_function
            if ms_flag:
                continue
            if minimum_non_msi_mem == previous_mem:
                self.update_performance_profile(msi_function.name, previous_mem)
                optimized_blocks.append(msi_function)
                if msi_function.copied_from is not None:
                    for function in self.petri_model.transitions:
                        if isinstance(function, Function) and function.copied_from == msi_function.copied_from:
                            optimized_blocks.append(function)
                skip_dfs_flag = False
            else:
                self.update_performance_profile(msi_function.name, minimum_non_msi_mem)
                if constraint_previously_satisfied_flag:
                    skip_dfs_flag = True
                else:
                    skip_dfs_flag = False
        for function in functions_to_be_revisited:
            ms_flag = False
            previous_mem = function.mem
            eligible_mem_options = {key: self.performance_profile_avg_rt_map[function.name][key] for key in
                                    self.performance_profile_avg_rt_map[function.name].keys() if
                                    key not in ineligible_mem_options_map[function.name] and
                                    self.performance_profile_avg_rt_map[function.name][key] <
                                    self.performance_profile_avg_rt_map[function.name][previous_mem]
                                    }
            sorted_mem_options = sorted(eligible_mem_options, key=eligible_mem_options.get)
            if len(sorted_mem_options) == 0:
                continue
            for i in range(len(sorted_mem_options)):
                mem = sorted_mem_options[i]
                self.update_performance_profile(function.name, mem)
                self.profile_multiprocessing(k=10000)
                if percentile is None:
                    new_cost = np.mean(self.cost_list)
                else:
                    new_cost = np.percentile(a=self.cost_list, q=percentile)
                if new_cost * 1000000 > budget_constraint:
                    ineligible_mem_options_map[function.name].append(mem)
                else:
                    ms_flag = True
                    break
            if not ms_flag:
                self.update_performance_profile(function.name, previous_mem)
                self.profile_multiprocessing(k=10000)
        return {fun.name: fun.mem for fun in self.petri_model.functions}

    def optimize_bcpc(self, performance_constraint: float, percentile: int = None) -> dict:
        """
        Solve the BCPC problem.

        Args:
          performance_constraint: A floating number of the performance constraint (end-to-end response time in milliseconds).
          percentile: K-th percentile.

        Returns: A dictionary of the optimal memory configurations for serverless functions.

        """
        self.initialize_with_best_performance_configuration()
        optimized_blocks = []
        functions_to_be_revisited = []
        ineligible_mem_options_map = {key: [] for key in self.function_names}
        early_reject_map = {key: float('inf') for key in self.function_names}
        skip_dfs_flag = False
        next_msi_function = None
        while True:
            if not skip_dfs_flag:
                msi_function = self.get_most_significant_cost_impact_function(parent_structure=None, check_visited=True,
                                                                              optimized_blocks=optimized_blocks)
                skip_dfs_flag = False
            else:
                msi_function = next_msi_function
            if msi_function is None:
                if functions_to_be_revisited is None:
                    configurations = {fun.name: fun.mem for fun in self.petri_model.functions}
                    return configurations
                else:
                    break
            previous_mem = msi_function.mem
            eligible_mem_options = {key: self.performance_profile_avg_cost_map[msi_function.name][key] for key in
                                    self.performance_profile_avg_cost_map[msi_function.name].keys() if
                                    key not in ineligible_mem_options_map[msi_function.name] and
                                    self.performance_profile_avg_cost_map[msi_function.name][key] <
                                    self.performance_profile_avg_cost_map[msi_function.name][previous_mem]
                                    }
            sorted_mem_options = sorted(eligible_mem_options, key=eligible_mem_options.get)
            if len(sorted_mem_options) == 0:
                optimized_blocks.append(msi_function)
                if msi_function.copied_from is not None:
                    for function in self.petri_model.transitions:
                        if isinstance(function, Function) and function.copied_from == msi_function.copied_from:
                            optimized_blocks.append(function)
                skip_dfs_flag = False
                continue
            constraint_previously_satisfied_flag = False
            ms_flag = False
            minimum_non_msi_ert = float('inf')
            minimum_non_msi_mem = previous_mem
            for i in range(len(sorted_mem_options)):
                mem = sorted_mem_options[i]
                if self.performance_profile_avg_rt_map[msi_function.name][mem] > early_reject_map[
                    msi_function.name] * self.early_reject_threshold:
                    continue
                self.update_performance_profile(msi_function.name, mem)
                self.profile_multiprocessing(k=10000)
                if percentile is None:
                    new_ert = np.mean(self.ert_list)
                else:
                    new_ert = np.percentile(a=self.ert_list, q=percentile)
                if new_ert > performance_constraint:
                    ineligible_mem_options_map[msi_function.name].append(mem)
                    if self.performance_profile_avg_rt_map[msi_function.name][mem] < early_reject_map[
                        msi_function.name]:
                        early_reject_map[msi_function.name] = self.performance_profile_avg_rt_map[msi_function.name][
                            mem]
                    continue
                new_msi_function = self.get_most_significant_cost_impact_function(parent_structure=None,
                                                                                  check_visited=True,
                                                                                  optimized_blocks=optimized_blocks)
                if new_msi_function == msi_function or (
                        new_msi_function.copied_from is not None
                        and new_msi_function.copied_from == msi_function.copied_from):
                    if not constraint_previously_satisfied_flag:
                        minimum_non_msi_mem = mem
                        optimized_blocks.append(msi_function)
                        if msi_function.copied_from is not None:
                            for function in self.petri_model.transitions:
                                if isinstance(function, Function) and function.copied_from == msi_function.copied_from:
                                    optimized_blocks.append(function)
                        if self.best_performance_configuration[msi_function.name] == previous_mem:
                            rt_cost_ratio = {m: self.performance_profile_avg_rt_cost_ratio_map_based_on_maximum_rt[
                                msi_function.name][m] for m in eligible_mem_options.keys() if
                                             self.performance_profile_avg_rt_cost_ratio_map_based_on_maximum_rt[
                                                 msi_function.name][m] > 0 and
                                             self.performance_profile_avg_rt_map[msi_function.name][m] <
                                             self.performance_profile_avg_rt_map[msi_function.name][
                                                 minimum_non_msi_mem] and
                                             self.performance_profile_avg_rt_cost_ratio_map_based_on_maximum_rt[
                                                 msi_function.name][m] >
                                             self.performance_profile_avg_rt_cost_ratio_map_based_on_maximum_rt[
                                                 msi_function.name][minimum_non_msi_mem]
                                             }
                            sorted_optimal_mem_options = sorted(rt_cost_ratio, key=rt_cost_ratio.get,
                                                                reverse=True)
                            rt_cost_ratio = {m: self.performance_profile_avg_rt_cost_ratio_map_based_on_maximum_rt[
                                msi_function.name][m] for m in eligible_mem_options.keys() if
                                             self.performance_profile_avg_rt_cost_ratio_map_based_on_maximum_rt[
                                                 msi_function.name][m] < 0 and
                                             self.performance_profile_avg_rt_map[msi_function.name][m] <
                                             self.performance_profile_avg_rt_map[msi_function.name][
                                                 minimum_non_msi_mem] and
                                             self.performance_profile_avg_rt_cost_ratio_map_based_on_maximum_rt[
                                                 msi_function.name][m] <
                                             self.performance_profile_avg_rt_cost_ratio_map_based_on_maximum_rt[
                                                 msi_function.name][minimum_non_msi_mem]
                                             }
                            sorted_optimal_mem_options += sorted(rt_cost_ratio, key=rt_cost_ratio.get,
                                                                 reverse=False)
                            if len(sorted_optimal_mem_options) == 0:
                                break
                            functions_to_be_revisited.append(msi_function)
                            skip_dfs_flag = False
                            for ms in sorted_optimal_mem_options:
                                if self.performance_profile_avg_rt_map[msi_function.name][ms] > early_reject_map[
                                    msi_function.name] * self.early_reject_threshold:
                                    continue
                                self.update_performance_profile(msi_function.name, ms)
                                self.profile_multiprocessing(k=10000)
                                if percentile is None:
                                    new_ert = np.mean(self.ert_list)
                                else:
                                    new_ert = np.percentile(a=self.ert_list, q=percentile)
                                if new_ert > performance_constraint:
                                    ineligible_mem_options_map[msi_function.name].append(ms)
                                    if self.performance_profile_avg_rt_map[msi_function.name][mem] < early_reject_map[
                                        msi_function.name]:
                                        early_reject_map[msi_function.name] = \
                                            self.performance_profile_avg_rt_map[msi_function.name][mem]
                                    continue
                                else:
                                    ms_flag = True
                                    break
                            if not ms_flag:
                                self.update_performance_profile(msi_function.name, mem)
                                self.profile_multiprocessing(k=10000)
                        break
                    else:
                        break
                else:
                    constraint_previously_satisfied_flag = True
                    if new_ert < minimum_non_msi_ert:
                        minimum_non_msi_ert = new_ert
                        minimum_non_msi_mem = mem
                        next_msi_function = new_msi_function
            if ms_flag:
                continue
            if minimum_non_msi_mem == previous_mem:
                self.update_performance_profile(msi_function.name, previous_mem)
                optimized_blocks.append(msi_function)
                if msi_function.copied_from is not None:
                    for function in self.petri_model.transitions:
                        if isinstance(function, Function) and function.copied_from == msi_function.copied_from:
                            optimized_blocks.append(function)
                skip_dfs_flag = False
            else:
                self.update_performance_profile(msi_function.name, minimum_non_msi_mem)
                if constraint_previously_satisfied_flag:
                    skip_dfs_flag = True
                else:
                    skip_dfs_flag = False
        for function in functions_to_be_revisited:
            ms_flag = False
            previous_mem = function.mem
            eligible_mem_options = {key: self.performance_profile_avg_cost_map[function.name][key] for key in
                                    self.performance_profile_avg_cost_map[function.name].keys() if
                                    key not in ineligible_mem_options_map[function.name] and
                                    self.performance_profile_avg_cost_map[function.name][key] <
                                    self.performance_profile_avg_cost_map[function.name][previous_mem]
                                    }
            sorted_mem_options = sorted(eligible_mem_options, key=eligible_mem_options.get)
            if len(sorted_mem_options) == 0:
                continue
            for i in range(len(sorted_mem_options)):
                mem = sorted_mem_options[i]
                self.update_performance_profile(function.name, mem)
                self.profile_multiprocessing(k=10000)
                if percentile is None:
                    new_ert = np.mean(self.ert_list)
                else:
                    new_ert = np.percentile(a=self.ert_list, q=percentile)
                if new_ert > performance_constraint:
                    ineligible_mem_options_map[function.name].append(mem)
                else:
                    ms_flag = True
                    break
            if not ms_flag:
                self.update_performance_profile(function.name, previous_mem)
                self.profile_multiprocessing(k=10000)
        return {fun.name: fun.mem for fun in self.petri_model.functions}
