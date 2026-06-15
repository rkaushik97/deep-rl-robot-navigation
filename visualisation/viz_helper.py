#!/usr/bin/env python3
"""Viz helper for the SAC inference demo (RViz / Foxglove).
Uses GROUND-TRUTH pose so everything is in a single 'world' frame (no odom drift):
broadcasts TF world->base_footprint->base_scan, a RED goal sphere, robot arrow, path,
floating 'Episode N', and /viz/distance_to_goal + /viz/episode for plots."""
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped, Quaternion, Twist
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32, Int32, ColorRGBA
from rviz_2d_overlay_msgs.msg import OverlayText
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

WORLD = "world"

def yaw_of(q):
    return math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))

def quat_yaw(yaw):
    return Quaternion(z=math.sin(yaw/2.0), w=math.cos(yaw/2.0))

class Viz(Node):
    def __init__(self):
        super().__init__("drl_viz_helper")
        gqos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Pose, "drl_goal_pose", self.on_goal, gqos)
        self.create_subscription(Odometry, "ground_truth_odom", self.on_gt, qos_profile_sensor_data)
        self.create_subscription(Odometry, "odom", self.on_odom, qos_profile_sensor_data)
        self.create_subscription(Twist, "cmd_vel", self.on_cmd, 10)
        self.goal_mk = self.create_publisher(Marker, "/viz/goal_marker", 1)
        self.txt_mk  = self.create_publisher(Marker, "/viz/episode_text", 1)
        self.robot_mk= self.create_publisher(Marker, "/viz/robot_marker", 1)
        self.path_pub= self.create_publisher(Path, "/viz/path", 1)
        self.dist_pub= self.create_publisher(Float32, "/viz/distance_to_goal", 10)
        self.ep_pub  = self.create_publisher(Int32, "/viz/episode", 10)
        self.hud_pub = self.create_publisher(OverlayText, "/viz/hud", 1)
        self.stats_pub = self.create_publisher(Marker, "/viz/stats", 1)
        self.tfb = TransformBroadcaster(self)
        self.stf = StaticTransformBroadcaster(self)
        # static base_footprint -> base_scan (lidar ~0.15 m up)
        st = TransformStamped()
        st.header.frame_id = "base_footprint"; st.child_frame_id = "base_scan"
        st.transform.translation.z = 0.15; st.transform.rotation.w = 1.0
        self.stf.sendTransform(st)
        self.goal = None; self.x = 0.0; self.y = 0.0; self.q = None; self.episode = 0
        self.ox = 0.0; self.oy = 0.0; self.oq = None
        self.success = 0; self.failure = 0; self.min_dist = float("inf")
        self.steps = 0; self.ep_start = None; self.cur_dist = 0.0
        self.last_outcome = "-"; self.last_steps = 0; self.last_time = 0.0
        self.path = Path(); self.path.header.frame_id = WORLD
        self.create_timer(0.1, self.tick)

    def on_goal(self, m):
        nx, ny = m.position.x, m.position.y
        now = self.get_clock().now()
        if self.goal is None:
            self.goal = (nx, ny); self.episode = 1; self.min_dist = float("inf")
            self.steps = 0; self.ep_start = now; self.path.poses = []; return
        if math.hypot(nx-self.goal[0], ny-self.goal[1]) > 0.4:
            ok = self.min_dist < 0.30   # reached the goal = success, else failure
            if ok: self.success += 1
            else: self.failure += 1
            self.last_outcome = "SUCCESS" if ok else "FAIL"
            self.last_steps = self.steps
            lt = (now - self.ep_start).nanoseconds / 1e9 if self.ep_start else 0.0
            self.last_time = lt if 0.0 <= lt <= 600.0 else 0.0
            self.episode += 1; self.min_dist = float("inf")
            self.steps = 0; self.ep_start = now; self.path.poses = []
        self.goal = (nx, ny)

    def on_gt(self, m):
        self.x = m.pose.pose.position.x; self.y = m.pose.pose.position.y; self.q = m.pose.pose.orientation

    def on_odom(self, m):
        self.ox = m.pose.pose.position.x; self.oy = m.pose.pose.position.y; self.oq = m.pose.pose.orientation

    def on_cmd(self, m):
        self.steps += 1

    def tick(self):
        now = self.get_clock().now().to_msg()
        # TF world -> odom (drift correction): world_T_odom = world_T_base * inv(odom_T_base)
        if self.q is not None and self.oq is not None:
            tyaw = yaw_of(self.q) - yaw_of(self.oq)
            c, s = math.cos(tyaw), math.sin(tyaw)
            t = TransformStamped(); t.header.stamp = now; t.header.frame_id = WORLD; t.child_frame_id = "odom"
            t.transform.translation.x = self.x - (c*self.ox - s*self.oy)
            t.transform.translation.y = self.y - (s*self.ox + c*self.oy)
            t.transform.rotation = quat_yaw(tyaw)
            self.tfb.sendTransform(t)
        self.ep_pub.publish(Int32(data=self.episode))
        # top HUD: algorithm + success/failure counters
        nowt = self.get_clock().now()
        if self.ep_start is None: self.ep_start = nowt
        elapsed = (nowt - self.ep_start).nanoseconds / 1e9
        if elapsed > 600.0 or elapsed < 0.0:   # ep_start was set before sim clock was ready
            self.ep_start = nowt; elapsed = 0.0
        tot = self.success + self.failure
        rate = (100.0 * self.success / tot) if tot else 0.0
        o = OverlayText(); o.action = OverlayText.ADD
        o.width = 620; o.height = 132
        o.horizontal_alignment = OverlayText.CENTER; o.vertical_alignment = OverlayText.TOP
        o.horizontal_distance = 0; o.vertical_distance = 12
        o.text_size = 16.0; o.font = "DejaVu Sans"
        o.fg_color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        o.bg_color = ColorRGBA(r=0.10, g=0.11, b=0.14, a=0.55)
        o.text = (f"SAC  —  inference (best checkpoint)        Episode {self.episode}\n"
                  f"Success {self.success}   /   Failure {self.failure}      ({rate:.0f}% success)\n"
                  f"this episode:   steps {self.steps}    time {elapsed:.1f}s    dist {self.cur_dist:.2f} m\n"
                  f"last episode:   {self.last_outcome}   ({self.last_steps} steps, {self.last_time:.1f}s)")
        self.hud_pub.publish(o)
        # compact stats as a TEXT marker, anchored at the top of the grid
        sm = Marker(); sm.header.frame_id = WORLD; sm.header.stamp = now; sm.ns = "stats"; sm.id = 0
        sm.type = Marker.TEXT_VIEW_FACING; sm.action = Marker.ADD
        sm.pose.position.x = 3.6; sm.pose.position.y = 0.0; sm.pose.position.z = 0.3; sm.pose.orientation.w = 1.0
        sm.scale.z = 0.16; sm.color = ColorRGBA(r=1.0, g=0.92, b=0.25, a=1.0)
        sm.text = ("Algorithm: SAC + reward V\n"
                   f"Episode: {self.episode}, Steps: {self.steps}, Time: {elapsed:.1f} secs\n"
                   f"Success: {self.success}, Failure: {self.failure}, Rate: {rate:.0f}%")
        self.stats_pub.publish(sm)
        # robot arrow
        rm = Marker(); rm.header.frame_id = WORLD; rm.header.stamp = now; rm.ns = "robot"; rm.type = Marker.ARROW
        rm.pose.position.x = self.x; rm.pose.position.y = self.y; rm.pose.position.z = 0.06
        rm.pose.orientation = self.q if self.q else Pose().orientation
        if not self.q: rm.pose.orientation.w = 1.0
        rm.scale.x = 0.35; rm.scale.y = 0.09; rm.scale.z = 0.09; rm.color = ColorRGBA(r=0.12, g=0.45, b=0.95, a=1.0)
        self.robot_mk.publish(rm)
        # path
        ps = PoseStamped(); ps.header.frame_id = WORLD; ps.header.stamp = now
        ps.pose.position.x = self.x; ps.pose.position.y = self.y; ps.pose.orientation.w = 1.0
        self.path.poses.append(ps); self.path.poses = self.path.poses[-3000:]
        self.path.header.stamp = now; self.path_pub.publish(self.path)
        if self.goal is None: return
        gx, gy = self.goal
        gm = Marker(); gm.header.frame_id = WORLD; gm.header.stamp = now; gm.ns = "goal"; gm.type = Marker.SPHERE
        gm.pose.position.x = gx; gm.pose.position.y = gy; gm.pose.position.z = 0.15; gm.pose.orientation.w = 1.0
        gm.scale.x = gm.scale.y = gm.scale.z = 0.45; gm.color = ColorRGBA(r=0.92, g=0.10, b=0.10, a=0.95)
        self.goal_mk.publish(gm)
        tm = Marker(); tm.header.frame_id = WORLD; tm.header.stamp = now; tm.ns = "eptext"; tm.type = Marker.TEXT_VIEW_FACING
        tm.pose.position.x = gx; tm.pose.position.y = gy; tm.pose.position.z = 0.65; tm.pose.orientation.w = 1.0
        tm.scale.z = 0.3; tm.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0); tm.text = f"Episode {self.episode}"
        self.txt_mk.publish(tm)
        d = float(math.hypot(gx-self.x, gy-self.y)); self.min_dist = min(self.min_dist, d); self.cur_dist = d
        self.dist_pub.publish(Float32(data=d))

def main():
    rclpy.init(); n = Viz()
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    n.destroy_node(); rclpy.shutdown()

if __name__ == "__main__":
    main()
