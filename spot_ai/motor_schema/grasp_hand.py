def execute(node, args: dict) -> None:
    if not isinstance(args, dict):
        args = {}

    logger = node.get_logger()
    grip_strength = args.get("strength", "default")
    logger.info(f"GraspHand command received: strength={grip_strength}")
    if not node._call_trigger_service(node.close_gripper_client, "/close_gripper", "close_gripper"):
        logger.warning("Falling back because /close_gripper is unavailable")
