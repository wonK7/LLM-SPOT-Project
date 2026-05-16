from glob import glob
from setuptools import find_packages, setup

package_name = 'spot_ai'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Voice command pipeline for Webots Spot and real Spot bring-up',
    license='TODO: License declaration',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'gemini_node = spot_ai.geminiAPI:main',
            'voice_ai_node = spot_ai.voice_ai_pipeline:main',
            'voice_ai_webots_node = spot_ai.voice_ai_webots_node:main',
            'voice_ai_real_spot_node = spot_ai.voice_ai_real_spot_node:main',
            'chat_tts_node = spot_ai.chat_tts_node:main',
            'vision_caption_node = spot_ai.vision_caption_node:main',
            'multi_camera_vision_node = spot_ai.multi_camera_vision_node:main',
            'task_policy_node = spot_ai.task_policy_node:main',
            'safety_gate_node = spot_ai.safety_gate_node:main',
            'safety_gate_webots_node = spot_ai.safety_gate_webots_node:main',
            'safety_gate_real_spot_node = spot_ai.safety_gate_real_spot_node:main',
            'mic_input_node = spot_ai.mic_input_node:main',
            'wav_input_node = spot_ai.wav_input_node:main',
            'wav_input_watcher_node = spot_ai.wav_input_watcher_node:main',
            'schema_service_node = spot_ai.schema_service_node:main',
            'schema_service_webots_node = spot_ai.schema_service_webots_node:main',
            'schema_service_real_spot_node = spot_ai.schema_service_real_spot_node:main',
        ],
    },
)
