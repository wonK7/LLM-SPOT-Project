def execute(node, args: dict) -> None:
    if not isinstance(args, dict):
        args = {}

    reason = str(args.get("reason", "voice_command")).strip() or "voice_command"

    # Stop is always handled locally because it is the safest fallback we have.
    node.get_logger().info(f"Stopping current motion locally. reason={reason}")
    node._stop_motion()
