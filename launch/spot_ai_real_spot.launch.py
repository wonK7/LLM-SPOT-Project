from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    enable_mic = LaunchConfiguration("enable_mic")
    enable_wav_input = LaunchConfiguration("enable_wav_input")
    enable_wav_watcher = LaunchConfiguration("enable_wav_watcher")
    enable_chat_tts = LaunchConfiguration("enable_chat_tts")
    enable_multi_camera_vision = LaunchConfiguration("enable_multi_camera_vision")
    enable_single_camera_vision = LaunchConfiguration("enable_single_camera_vision")
    enable_tts = LaunchConfiguration("enable_tts")
    enable_arm = LaunchConfiguration("enable_arm")
    enable_gripper = LaunchConfiguration("enable_gripper")
    publish_point_clouds = LaunchConfiguration("publish_point_clouds")
    whisper_language = LaunchConfiguration("whisper_language")
    require_wake_word = LaunchConfiguration("require_wake_word")
    wav_audio_file = LaunchConfiguration("wav_audio_file")
    single_camera_image_topic = LaunchConfiguration("single_camera_image_topic")
    vision_query_topic = LaunchConfiguration("vision_query_topic")
    vision_output_topic = LaunchConfiguration("vision_output_topic")

    return LaunchDescription(
        [
            DeclareLaunchArgument("enable_mic", default_value="false"),
            DeclareLaunchArgument("enable_wav_input", default_value="false"),
            DeclareLaunchArgument("enable_wav_watcher", default_value="false"),
            DeclareLaunchArgument("enable_chat_tts", default_value="true"),
            DeclareLaunchArgument("enable_multi_camera_vision", default_value="false"),
            DeclareLaunchArgument("enable_single_camera_vision", default_value="false"),
            DeclareLaunchArgument("enable_tts", default_value="false"),
            DeclareLaunchArgument("enable_arm", default_value="true"),
            DeclareLaunchArgument("enable_gripper", default_value="true"),
            DeclareLaunchArgument("publish_point_clouds", default_value="true"),
            DeclareLaunchArgument("whisper_language", default_value="en"),
            DeclareLaunchArgument("require_wake_word", default_value="false"),
            DeclareLaunchArgument(
                "wav_audio_file",
                default_value="/tmp/spot_ai_input.wav",
            ),
            DeclareLaunchArgument(
                "single_camera_image_topic",
                default_value="/camera/frontleft/image",
            ),
            DeclareLaunchArgument(
                "vision_query_topic",
                default_value="/spot_ai/vision_query",
            ),
            DeclareLaunchArgument(
                "vision_output_topic",
                default_value="/spot_ai/chat_output",
            ),
            Node(
                package="spot_ai",
                executable="voice_ai_real_spot_node",
                name="voice_ai_real_spot_node",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": False,
                        "enable_tts": enable_tts,
                        "require_wake_word": require_wake_word,
                    }
                ],
            ),
            Node(
                package="spot_ai",
                executable="schema_service_real_spot_node",
                name="schema_service_real_spot_node",
                output="screen",
            ),
            Node(
                package="spot_ai",
                executable="safety_gate_real_spot_node",
                name="safety_gate_real_spot_node",
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
                executable="wav_input_node",
                name="wav_input_node",
                output="screen",
                condition=IfCondition(enable_wav_input),
                parameters=[
                    {
                        "audio_file": wav_audio_file,
                        "whisper_language": whisper_language,
                    }
                ],
            ),
            Node(
                package="spot_ai",
                executable="wav_input_watcher_node",
                name="wav_input_watcher_node",
                output="screen",
                condition=IfCondition(enable_wav_watcher),
                parameters=[
                    {
                        "audio_file": wav_audio_file,
                        "whisper_language": whisper_language,
                    }
                ],
            ),
            Node(
                package="spot_ai",
                executable="chat_tts_node",
                name="chat_tts_node",
                output="screen",
                condition=IfCondition(enable_chat_tts),
                parameters=[{"auto_play_windows": False}],
            ),
            Node(
                package="spot_ai",
                executable="multi_camera_vision_node",
                name="multi_camera_vision_node",
                output="screen",
                condition=IfCondition(enable_multi_camera_vision),
            ),
            Node(
                package="spot_ai",
                executable="vision_caption_node",
                name="vision_caption_node",
                output="screen",
                condition=IfCondition(enable_single_camera_vision),
                parameters=[
                    {
                        "image_topic": single_camera_image_topic,
                        "query_topic": vision_query_topic,
                        "output_topic": vision_output_topic,
                    }
                ],
            ),
        ]
    )
