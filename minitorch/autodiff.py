from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    lst_vals = list(vals)
    lst_vals[arg] += epsilon / 2
    nxt_val = f(*lst_vals)
    lst_vals[arg] -= epsilon
    prev_val = f(*lst_vals)
    return (nxt_val - prev_val) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    graph = {}
    id_variable_mapping = {variable.unique_id: variable}
    variable_ids = {variable.unique_id, }
    parent_variable_count = {variable.unique_id: 0}
    while len(variable_ids) > 0:
        root_variable = id_variable_mapping[variable_ids.pop()]
        if root_variable.unique_id not in graph:
            graph[root_variable.unique_id] = {}
        if root_variable.history and root_variable.history.inputs:
            for child_variable in root_variable.history.inputs:
                variable_ids.add(child_variable.unique_id)
                id_variable_mapping[child_variable.unique_id] = child_variable
                if child_variable.unique_id in graph[root_variable.unique_id]:
                    graph[root_variable.unique_id][child_variable.unique_id] += 1
                else:
                    graph[root_variable.unique_id][child_variable.unique_id] = 1
                if child_variable.unique_id in parent_variable_count:
                    parent_variable_count[child_variable.unique_id] += 1
                else:
                    parent_variable_count[child_variable.unique_id] = 1

    root_variable_ids = []
    for unique_id in parent_variable_count:
        if parent_variable_count[unique_id] == 0:
            root_variable_ids.append(unique_id)

    sorted_variables = []
    while len(root_variable_ids) > 0:
        root_variable = id_variable_mapping[root_variable_ids[0]]
        root_variable_ids = root_variable_ids[1:]
        sorted_variables.append(root_variable)
        if root_variable.unique_id in graph:
            for child_variable_id in graph[root_variable.unique_id]:
                # Note that one root node can have multi out degree to one child node
                parent_variable_count[child_variable_id] -= graph[root_variable.unique_id][child_variable_id]
                if parent_variable_count[child_variable_id] == 0:
                    root_variable_ids.append(child_variable_id)

    return sorted_variables


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    sorted_variables = topological_sort(variable)
    id_derivative_mapping = {variable.unique_id: deriv}
    for variable in sorted_variables:
        derivative = id_derivative_mapping[variable.unique_id]
        if variable.is_leaf():
            variable.accumulate_derivative(derivative)
        elif not variable.is_constant():
            back = variable.chain_rule(derivative)
            for child_variable, child_derivative in back:
                if child_variable.unique_id in id_derivative_mapping:
                    id_derivative_mapping[child_variable.unique_id] += child_derivative
                else:
                    id_derivative_mapping[child_variable.unique_id] = child_derivative


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
