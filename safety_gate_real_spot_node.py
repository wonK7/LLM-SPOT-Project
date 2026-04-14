from .adapter_profiles import REAL_SPOT_ADAPTER, get_adapter_defaults
from .safety_gate_node import main as safety_gate_main


def main(args=None) -> None:
    safety_gate_main(
        args=args,
        parameter_defaults=get_adapter_defaults(REAL_SPOT_ADAPTER, "safety_gate"),
    )
