def not_implemented(node, snippet_name: str) -> None:
    node.get_logger().warning(
        f"'{snippet_name}' is recognized, but the actual robot action is not ready yet. Stopping instead."
    )
    node._stop_motion()


def coerce_float(args: dict, key: str, default: float) -> float:
    if not isinstance(args, dict):
        return float(default)
    try:
        return float(args.get(key, default))
    except (TypeError, ValueError):
        return float(default)
