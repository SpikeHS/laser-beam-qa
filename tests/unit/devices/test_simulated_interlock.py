from lbqa_contracts.errors import ErrorCode
from lbqa_devices.safety.simulated_interlock import SimulatedInterlock


def test_simulated_interlock_unsafe_returns_false() -> None:
    interlock = SimulatedInterlock()

    interlock.set_unsafe("cover open")

    assert interlock.is_safe() is False
    status = interlock.get_status()
    assert status["safe"] is False
    assert status["reason"] == "cover open"
    assert status["error_code"] == ErrorCode.E_SAFETY_INTERLOCK_OPEN.value
