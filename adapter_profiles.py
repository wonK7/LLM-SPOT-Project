from copy import deepcopy


WEBOTS_ADAPTER = "webots"
REAL_SPOT_ADAPTER = "real_spot"


ADAPTER_PROFILES = {
    WEBOTS_ADAPTER: {
        "description": "Webots simulation adapter profile",
        "voice_ai": {
            "cmd_vel_topic": "/spot_ai/cmd_vel_raw",
            "odom_topic": "/Spot/odometry",
            "require_wake_word": False,
            "enable_heading_hold": True,
            "heading_hold_use_sim_only": True,
            "auto_sim_scale": True,
            "allow_windows_popup_fallback": False,
            "auto_play_windows": False,
        },
        "schema_service": {
            "output_cmd_topic": "/spot_ai/cmd_vel_raw",
            "gripper_cmd_topic": "/spot/gripper/command",
        },
        "safety_gate": {
            "input_cmd_topic": "/spot_ai/cmd_vel_raw",
            "output_cmd_topic": "/cmd_vel",
            "scan_topic": "/scan",
            "enabled": True,
            "deadman_timeout_sec": 0.35,
        },
        "mic": {
            "whisper_language": "auto",
            "record_duration_sec": 6.0,
            "record_interval_sec": 0.5,
        },
    },
    REAL_SPOT_ADAPTER: {
        "description": "Real Boston Dynamics Spot adapter profile",
        "voice_ai": {
            "cmd_vel_topic": "/spot_ai/cmd_vel_raw",
            "odom_topic": "/spot/odometry",
            "require_wake_word": False,
            "enable_heading_hold": False,
            "heading_hold_use_sim_only": False,
            "auto_sim_scale": False,
            "allow_windows_popup_fallback": False,
            "auto_play_windows": False,
        },
        "schema_service": {
            "output_cmd_topic": "/spot_ai/cmd_vel_raw",
            "gripper_cmd_topic": "/spot/gripper/command",
        },
        "safety_gate": {
            "input_cmd_topic": "/spot_ai/cmd_vel_raw",
            "output_cmd_topic": "/cmd_vel",
            "scan_topic": "/scan",
            "enabled": True,
            "deadman_timeout_sec": 0.35,
        },
        "mic": {
            "whisper_language": "en",
            "record_duration_sec": 6.0,
            "record_interval_sec": 0.5,
        },
    },
}


def get_adapter_defaults(adapter_name: str, component_name: str) -> dict:
    profile = ADAPTER_PROFILES.get(adapter_name)
    if not profile:
        raise ValueError(f"Unknown adapter profile: {adapter_name}")
    return deepcopy(profile.get(component_name, {}))
