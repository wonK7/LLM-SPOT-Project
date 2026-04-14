from .adapter_profiles import REAL_SPOT_ADAPTER, get_adapter_defaults
from .voice_ai_pipeline import main as voice_ai_main


def main(args=None) -> None:
    voice_ai_main(
        args=args,
        parameter_defaults=get_adapter_defaults(REAL_SPOT_ADAPTER, "voice_ai"),
    )
