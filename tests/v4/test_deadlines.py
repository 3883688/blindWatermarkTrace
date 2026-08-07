import pytest

from trace_app.v4.deadlines import Deadline, DeadlineExceeded


def test_deadline_children_cannot_extend_parent() -> None:
    now = [100.0]
    parent = Deadline.after(10, clock=lambda: now[0])

    assert parent.child(30).expires_at == parent.expires_at
    assert parent.child(3).expires_at == 103.0
    now[0] = 110.0
    with pytest.raises(DeadlineExceeded, match="geometry"):
        parent.check("geometry")


def test_approved_deadline_factories_are_capped() -> None:
    assert Deadline.synchronous(clock=lambda: 0.0).expires_at == 300.0
    assert Deadline.deep(clock=lambda: 0.0).expires_at == 1000.0
    with pytest.raises(ValueError):
        Deadline.after(301, maximum_seconds=300)
