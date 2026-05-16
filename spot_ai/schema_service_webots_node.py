from .adapter_profiles import WEBOTS_ADAPTER, get_adapter_defaults
from .schema_service_node import main as schema_service_main


def main(args=None) -> None:
    schema_service_main(
        args=args,
        parameter_defaults=get_adapter_defaults(WEBOTS_ADAPTER, "schema_service"),
    )
