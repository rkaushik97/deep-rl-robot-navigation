"""Lean stage-9 sim launch for DRL training.

Brings up only what training needs:
  - gz-sim server (headless-capable) running the stage-9 world
  - optional gz GUI client (skipped with headless:=true)
  - the robot spawned from its SDF
  - the ros_gz topic bridge (/scan, /odom, /ground_truth_odom, /cmd_vel, /clock, ...)
  - the ros_gz service bridge (world pause/unpause + entity spawn/remove/set_pose)

Dropped vs. the original: SLAM Toolbox, Nav2, RViz, path_publisher, gt_tf_publisher, and
robot_state_publisher. The environment/agent read /scan and /ground_truth_odom directly, so no
TF tree is needed for training. turtlebot3_common meshes are bundled in this package's models/,
so there is no turtlebot3_gazebo dependency.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
)
from launch.conditions import UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


STAGE = '9'
WORLD_NAME = 'drl_stage9'


def generate_launch_description():
    drl_gz_share = get_package_share_directory('turtlebot3_drl_gazebo')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    x_pose = LaunchConfiguration('x_pose', default='0.0')
    y_pose = LaunchConfiguration('y_pose', default='0.0')
    headless = LaunchConfiguration('headless', default='false')

    world = os.path.join(drl_gz_share, 'worlds', 'turtlebot3_drl_stage9.world')
    robot_sdf = os.path.join(drl_gz_share, 'models', 'drl_burger', 'model.sdf')
    bridge_yaml = os.path.join(drl_gz_share, 'params', 'drl_bridge.yaml')

    # Make our bundled models (drl_burger, turtlebot3_common meshes, world models) discoverable,
    # and the obstacle_animator system plugin loadable.
    set_drl_models = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(drl_gz_share, 'models'),
    )
    set_plugin_path = AppendEnvironmentVariable(
        'GZ_SIM_SYSTEM_PLUGIN_PATH',
        os.path.join(drl_gz_share, '..', '..', 'lib', 'turtlebot3_drl_gazebo'),
    )

    # The environment/agent read the active stage from this file (at import time).
    write_stage = ExecuteProcess(
        cmd=['bash', '-c', f'echo {STAGE} > /tmp/drlnav_current_stage.txt'],
        output='screen',
    )

    # gz server: -r run immediately, -s server-only, -v2 warnings
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': ['-r -s -v2 ', world]}.items(),
    )
    # gz GUI client: -g, skipped when headless:=true
    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-g -v2 '}.items(),
        condition=UnlessCondition(headless),
    )

    spawn_turtlebot_cmd = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'burger',
            '-file', robot_sdf,
            '-x', x_pose,
            '-y', y_pose,
            '-z', '0.01',
        ],
        output='screen',
    )

    bridge_cmd = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['--ros-args', '-p', f'config_file:={bridge_yaml}'],
        output='screen',
    )

    gz_service_bridge_args = [
        f'/world/{WORLD_NAME}/control@ros_gz_interfaces/srv/ControlWorld',
        f'/world/{WORLD_NAME}/create@ros_gz_interfaces/srv/SpawnEntity',
        f'/world/{WORLD_NAME}/remove@ros_gz_interfaces/srv/DeleteEntity',
        f'/world/{WORLD_NAME}/set_pose@ros_gz_interfaces/srv/SetEntityPose',
    ]
    service_bridge_cmd = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=gz_service_bridge_args,
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('headless', default_value='false',
                              description='Skip the gz GUI client (faster for training)'),
        set_drl_models,
        set_plugin_path,
        write_stage,
        gzserver_cmd,
        gzclient_cmd,
        spawn_turtlebot_cmd,
        bridge_cmd,
        service_bridge_cmd,
    ])
