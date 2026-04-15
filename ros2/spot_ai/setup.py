from setuptools import find_packages, setup

package_name = 'spot_ai'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hyewon',
    maintainer_email='hyewon@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'gemini_node = spot_ai.geminiAPI:main',
            'voice_ai_node = spot_ai.voice_ai_pipeline:main',
            'wav_input_node = spot_ai.wav_input_node:main',
            'chat_tts_node = spot_ai.chat_tts_node:main',
        ],
    },
)
