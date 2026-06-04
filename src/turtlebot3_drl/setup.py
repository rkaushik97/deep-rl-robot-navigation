from setuptools import find_packages, setup

package_name = 'turtlebot3_drl'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),                       # finds the inner turtlebot3_drl/ module
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kaushik',
    maintainer_email='kaushikraghupathruni@gmail.com',
    description='Single-agent DDPG trainer for TurtleBot3 (experiment-tracked).',
    license='Apache 2.0',
    entry_points={
        'console_scripts': [
            'environment = turtlebot3_drl.drl_environment.drl_environment:main',
            'gazebo_goals = turtlebot3_drl.drl_gazebo.drl_gazebo:main',
            'train_agent = turtlebot3_drl.drl_agent.drl_agent:main_train',
            'test_agent = turtlebot3_drl.drl_agent.drl_agent:main_test',
        ],
    },
)
