def execute(node, args: dict) -> None:
    del args
    node._stop_motion()
    if not node._call_trigger_service(node.sit_client, "/sit", "sit"):
        node.get_logger().warning("Falling back to stop because /sit is unavailable")
