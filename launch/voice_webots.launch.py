from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    enable_mic = LaunchConfiguration('enable_mic')
    enable_chat_tts = LaunchConfiguration('enable_chat_tts')
    enable_tts = LaunchConfiguration('enable_tts')
    require_wake_word = LaunchConfiguration('require_wake_word')
    whisper_language = LaunchConfiguration('whisper_language')

    voice_ai_node = Node(
        package='spot_ai',
        executable='voice_ai_node',
        name='voice_ai_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'enable_tts': enable_tts,
            'cmd_vel_topic': '/spot_ai/cmd_vel_raw',
            'odom_topic': '/Spot/odometry',
            'require_wake_word': require_wake_word,
        }],
    )

    safety_gate_node = Node(
        package='spot_ai',
        executable='safety_gate_node',
        name='safety_gate_node',
        output='screen',
        parameters=[{
            'input_cmd_topic': '/spot_ai/cmd_vel_raw',
            'output_cmd_topic': '/cmd_vel',
            'scan_topic': '/scan',
            'enabled': True,
        }],
    )

    mic_input_node = Node(
        package='spot_ai',
        executable='mic_input_node',
        name='mic_input_node',
        output='screen',
        condition=IfCondition(enable_mic),
        parameters=[{
            'whisper_language': whisper_language,
        }],
    )

    chat_tts_node = Node(
        package='spot_ai',
        executable='chat_tts_node',
        name='chat_tts_node',
        output='screen',
        condition=IfCondition(enable_chat_tts),
        parameters=[{'auto_play_windows': True}],
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('enable_mic', default_value='false'),
        DeclareLaunchArgument('enable_chat_tts', default_value='true'),
        DeclareLaunchArgument('enable_tts', default_value='false'),
        DeclareLaunchArgument('require_wake_word', default_value='false'),
        DeclareLaunchArgument('whisper_language', default_value='auto'),
        voice_ai_node,
        safety_gate_node,
        mic_input_node,
        chat_tts_node,
    ])
