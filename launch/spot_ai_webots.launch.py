from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    enable_mic = LaunchConfiguration("enable_mic")
    enable_chat_tts = LaunchConfiguration("enable_chat_tts")
    enable_tts = LaunchConfiguration("enable_tts")
    whisper_language = LaunchConfiguration("whisper_language")
    require_wake_word = LaunchConfiguration("require_wake_word")

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("enable_mic", default_value="false"),
            DeclareLaunchArgument("enable_chat_tts", default_value="true"),
            DeclareLaunchArgument("enable_tts", default_value="false"),
            DeclareLaunchArgument("whisper_language", default_value="auto"),
            DeclareLaunchArgument("require_wake_word", default_value="false"),
            Node(
                package="spot_ai",
                executable="voice_ai_webots_node",
                name="voice_ai_webots_node",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "enable_tts": enable_tts,
                        "require_wake_word": require_wake_word,
                    }
                ],
            ),
            Node(
                package="spot_ai",
                executable="schema_service_webots_node",
                name="schema_service_webots_node",
                output="screen",
            ),
            Node(
                package="spot_ai",
                executable="safety_gate_webots_node",
                name="safety_gate_webots_node",
                output="screen",
            ),
            Node(
                package="spot_ai",
                executable="mic_input_node",
                name="mic_input_node",
                output="screen",
                condition=IfCondition(enable_mic),
                parameters=[{"whisper_language": whisper_language}],
            ),
            Node(
                package="spot_ai",
                executable="chat_tts_node",
                name="chat_tts_node",
                output="screen",
                condition=IfCondition(enable_chat_tts),
                parameters=[{"auto_play_windows": False}],
            ),
        ]
    )
