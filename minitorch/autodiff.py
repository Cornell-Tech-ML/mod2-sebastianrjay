from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    val_list = list(vals)
    x_plus_e_vals = val_list[0:arg] + [vals[arg] + epsilon] + val_list[arg + 1 :]
    x_minus_e_vals = val_list[0:arg] + [vals[arg] - epsilon] + val_list[arg + 1 :]
    return (f(*x_plus_e_vals) - f(*x_minus_e_vals)) / (2 * epsilon)


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative of the variable with respect to the
        computation graph output.
        """
        ...

    @property
    def unique_id(self) -> int:
        """Returns a unique identifier of the variable."""
        ...

    def is_leaf(self) -> bool:
        """True if this variable has no incoming edges in the computation graph."""
        ...

    def is_constant(self) -> bool:
        """True if this variable is a constant (i.e. not a learned parameter)."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parent variables in the computation graph."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to backpropagate the gradients of the output relative to the input."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    result, perm_visit, temp_visit = [], set(), set()

    # NOTE: the snippet below matches the DFS topological sort algorithm
    # shown on Wikipedia:
    # https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search
    def visit(variable: Variable) -> None:
        if variable.is_constant():
            return
        if variable.unique_id in perm_visit:
            return
        if variable.unique_id in temp_visit:
            raise ValueError("Cycle detected")
        temp_visit.add(variable.unique_id)
        for parent in variable.parents:
            visit(parent)
        temp_visit.remove(variable.unique_id)
        perm_visit.add(variable.unique_id)
        result.append(variable)

    visit(variable)
    return list(reversed(result))  # Reversed since array is used instead of linked list


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable.
        deriv: Its derivative that we want to propagate backward to the leaves.

    Returns:
    -------
        No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    variables = topological_sort(variable)
    derivs = {}
    derivs[variable.unique_id] = deriv
    for var in variables:
        if var.is_constant():
            continue
        curr_deriv = derivs.get(var.unique_id, 0)
        if var.is_leaf():
            var.accumulate_derivative(curr_deriv)
            derivs[var.unique_id] = derivs.get(var.unique_id, 0) + var.derivative  # pyright: ignore
        else:
            chain_rule_output = var.chain_rule(curr_deriv)
            for scalar, d_out in chain_rule_output:
                derivs[scalar.unique_id] = derivs.get(scalar.unique_id, 0) + d_out


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved values set above, to be used during backpropagation."""
        return self.saved_values
