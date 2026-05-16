def execute(node, args: dict) -> None:
    if not isinstance(args, dict):
        args = {}

    logger = node.get_logger()
    release_mode = args.get("mode", "default")
    logger.info(f"ReleaseHand command received: mode={release_mode}")
    if not node._call_trigger_service(node.open_gripper_client, "/open_gripper", "open_gripper"):
        logger.warning("Falling back because /open_gripper is unavailable")
