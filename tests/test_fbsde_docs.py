import fbsde
from fbsde.lq import riccati_solution, solve_lq_fbsde


def test_riccati_solution_docstring():
    doc = riccati_solution.__doc__
    assert doc is not None
    assert "Riccati" in doc
    assert "-\\dot{P}(t)" in doc


def test_solve_lq_fbsde_docstring():
    doc = solve_lq_fbsde.__doc__
    assert doc is not None
    assert "single path" in doc
    assert "(N+1,)" in doc
