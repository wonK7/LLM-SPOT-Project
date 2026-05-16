from .walk_forward import execute as execute_walk_forward
from .stop import execute as execute_stop
from .walk_backward import execute as execute_walk_backward
from .walk_left import execute as execute_walk_left
from .walk_right import execute as execute_walk_right
from .rotate_left import execute as execute_rotate_left
from .rotate_right import execute as execute_rotate_right
from .stand import execute as execute_stand
from .sit import execute as execute_sit
from .arm_unstow import execute as execute_arm_unstow
from .arm_stow import execute as execute_arm_stow
from .grasp_hand import execute as execute_grasp_hand
from .release_hand import execute as execute_release_hand
from .extend_arm import execute as execute_extend_arm

MOTOR_SCHEMA_EXECUTORS = {
    "walk_forward": execute_walk_forward,
    "stop": execute_stop,
    "walk_backward": execute_walk_backward,
    "walk_left": execute_walk_left,
    "walk_right": execute_walk_right,
    "rotate_left": execute_rotate_left,
    "rotate_right": execute_rotate_right,
    "stand": execute_stand,
    "sit": execute_sit,
    "arm_unstow": execute_arm_unstow,
    "arm_stow": execute_arm_stow,
    "grasp_hand": execute_grasp_hand,
    "release_hand": execute_release_hand,
    "extend_arm": execute_extend_arm,
}
