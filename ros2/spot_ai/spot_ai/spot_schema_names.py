SCHEMA_COMMANDS = [
    "WalkForward",
    "WalkBackward",
    "WalkLeft",
    "WalkRight",
    "Stop",
    "RotateLeft",
    "RotateRight",
    "Stand",
    "Sit",
    "GraspHand",
    "ReleaseHand",
    "ExtendArm",
]

PRIMITIVE_TO_SNIPPET = {
    "WalkForward": "walk_forward",
    "WalkBackward": "walk_backward",
    "WalkLeft": "walk_left",
    "WalkRight": "walk_right",
    "Stop": "stop",
    "RotateLeft": "rotate_left",
    "RotateRight": "rotate_right",
    "Stand": "stand",
    "Sit": "sit",
    "GraspHand": "grasp_hand",
    "ReleaseHand": "release_hand",
    "ExtendArm": "extend_arm",
}

SUPPORTED_EXECUTION_SNIPPETS = {
    "walk_forward",
    "stop",
}
