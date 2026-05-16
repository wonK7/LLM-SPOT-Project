from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='spot_ai',
            executable='mic_input_node',
            name='mic_input_node',
            output='screen',
        ),
    ])
