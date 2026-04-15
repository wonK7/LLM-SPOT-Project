from spot_ai.snippets.stop import execute as stop_execute


def _walk_forward(node, args: dict) -> None:
    distance_m = float(args.get("distance_m", 1.0))
    speed_mps = float(args.get("speed_mps", 0.45))
    node._start_walk_forward(distance_m, speed_mps)


SNIPPET_EXECUTORS = {
    "walk_forward": _walk_forward,
    "stop": stop_execute,
}
