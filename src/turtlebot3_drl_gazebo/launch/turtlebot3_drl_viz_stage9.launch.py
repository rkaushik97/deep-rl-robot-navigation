"""Stage-9 sim launch WITH visualization, for running inference on a workstation.

This is the lean training launch (turtlebot3_drl_stage9.launch.py) plus the GUI bits that
were intentionally left out of the trainer, each opt-in via a launch arg:

  headless:=false   show the gz-sim GUI client (the Gazebo window). Default false here.
  rviz:=true        start RViz2 with the bundled config + robot_state_publisher so the
                    burger mesh, /scan, /odom and TF render. Default false.
  capture_cam:=true spawn an overhead camera publishing /capture_cam and bridge it to ROS,
                    for scripts/capture_visual.py (the network-activations demo). Default false.

Intended for the :viz image (rviz2 + ros_gz_image installed) driven over X11 forwarding,
e.g. XQuartz on macOS. The training world is reused untouched — the capture camera is spawned
as a separate entity at runtime so it never weighs down headless training renders.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


STAGE = '9'


def generate_launch_description():
    drl_gz_share = get_package_share_directory('turtlebot3_drl_gazebo')

    headless = LaunchConfiguration('headless', default='false')
    use_rviz = LaunchConfiguration('rviz', default='false')
    # robot_state_publisher can be started independently of RViz so an externally-launched
    # RViz (e.g. on a different display over X11 forwarding) still gets /robot_description + TF.
    use_rsp = LaunchConfiguration('robot_state_pub', default='false')
    use_capture = LaunchConfiguration('capture_cam', default='false')

    rviz_config = os.path.join(drl_gz_share, 'rviz', 'drl.rviz')
    capture_cam_sdf = os.path.join(drl_gz_share, 'models', 'capture_cam', 'model.sdf')

    # Read the burger URDF so RViz's RobotModel display has a description + link TF.
    tb3_desc = get_package_share_directory('turtlebot3_description')
    urdf_path = os.path.join(tb3_desc, 'urdf', 'turtlebot3_burger.urdf')
    with open(urdf_path, 'r') as f:
        robot_description = f.read()

    # --- the lean stage-9 stack (gz server + robot + bridges), GUI client gated by headless ---
    base_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(drl_gz_share, 'launch', 'turtlebot3_drl_stage9.launch.py')
        ),
        launch_arguments={'headless': headless}.items(),
    )

    # --- RViz path: robot_state_publisher (mesh + link TF) + RViz2 with the bundled config ---
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
        condition=IfCondition(use_rsp),
    )
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(use_rviz),
    )

    # --- capture-cam path: spawn the overhead camera, then bridge its image topic to ROS ---
    # Delayed so the gz server (started by base_stack) is up before we spawn into it.
    spawn_capture_cam = TimerAction(
        period=20.0,
        actions=[Node(
            package='ros_gz_sim',
            executable='create',
            arguments=['-name', 'capture_cam', '-file', capture_cam_sdf,
                       '-x', '0', '-y', '0', '-z', '6',
                       '-R', '0', '-P', '1.5708', '-Y', '1.5708'],
            output='screen',
            condition=IfCondition(use_capture),
        )],
    )
    capture_bridge = Node(
        package='ros_gz_image',
        executable='image_bridge',
        arguments=['/capture_cam'],
        output='screen',
        condition=IfCondition(use_capture),
    )

    return LaunchDescription([
        DeclareLaunchArgument('headless', default_value='false',
                              description='Skip the gz GUI client'),
        DeclareLaunchArgument('rviz', default_value='false',
                              description='Start RViz2 (in-launch, uses this process display)'),
        DeclareLaunchArgument('robot_state_pub', default_value='false',
                              description='Start robot_state_publisher (URDF + link TF for RViz)'),
        DeclareLaunchArgument('capture_cam', default_value='false',
                              description='Spawn the overhead /capture_cam camera + image bridge'),
        base_stack,
        robot_state_publisher,
        rviz,
        spawn_capture_cam,
        capture_bridge,
    ])
