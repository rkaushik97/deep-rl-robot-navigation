#!/usr/bin/env python3
# Run a trained DDPG actor through episodes; per step capture the 44-dim state, both hidden
# activations (h0=relu(fa1), h1=relu(fa2)), the action, accumulated reward, AND the gazebo
# camera frame (/capture_cam). Saves the first SUCCESS episode to an .npz for the side-by-side
# (gazebo + network) visualization.
#   python3 scripts/capture_visual.py <actor.pt> <out.npz>
#   Needs an overhead camera publishing /capture_cam. Add this to the stage9 world before
#   capturing (removed afterwards to keep training renders light):
#     <model name="capture_cam"><static>true</static><pose>0 0 6 0 1.5708 1.5708</pose><link name="link">
#       <sensor name="cam" type="camera"><update_rate>20</update_rate><topic>capture_cam</topic>
#       <camera><horizontal_fov>1.2</horizontal_fov><image><width>600</width><height>600</height></image>
#       <clip><near>0.1</near><far>40</far></clip></camera></sensor></link></model>
#   Bridge it:  ros2 run ros_gz_image image_bridge /capture_cam
import sys, time, copy
import numpy as np, torch, rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from turtlebot3_msgs.srv import DrlStep, Goal
from ros_gz_interfaces.srv import ControlWorld
from turtlebot3_drl.common import utilities as util
from turtlebot3_drl.algorithms.ddpg.ddpg import DDPG
from turtlebot3_drl.common.settings import SUCCESS


class Capture(Node):
    def __init__(self, ckpt, out):
        super().__init__('capture_visual')
        self.real_robot = 0
        self.device = torch.device('cpu')
        self.model = DDPG(self.device, 1)
        self.model.actor.load_state_dict(torch.load(ckpt, map_location='cpu'))
        self.model.actor.eval()
        self.step_comm_client = self.create_client(DrlStep, 'step_comm')
        self.goal_comm_client = self.create_client(Goal, 'goal_comm')
        self.gazebo_control = self.create_client(ControlWorld, f'/world/drl_stage{util.stage}/control')
        self.frame = None
        self.create_subscription(Image, '/capture_cam', self._img, 10)
        self.run(out)

    def _img(self, m):
        self.frame = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)[..., :3].copy()

    def fwd(self, state):
        s = torch.from_numpy(np.asarray(state, np.float32))
        if s.dim() == 1:
            s = s.unsqueeze(0)
        with torch.no_grad():
            h0 = torch.relu(self.model.actor.fa1(s))
            h1 = torch.relu(self.model.actor.fa2(h0))
            a = torch.tanh(self.model.actor.fa3(h1))
        return h0.squeeze(0).numpy(), h1.squeeze(0).numpy(), a.squeeze(0).numpy()

    def run(self, out):
        util.pause_simulation(self, 0)
        for ep in range(25):
            util.wait_new_goal(self)
            state = util.init_episode(self)
            ap = [0.0, 0.0]
            util.unpause_simulation(self, 0); time.sleep(0.5)
            done = False; outcome = 0; cum = 0.0
            S, H0, H1, A, CR, F = [], [], [], [], [], []
            while not done:
                h0, h1, a = self.fwd(state)
                ns, reward, done, outcome, _, _ = util.step(self, a.tolist(), ap)
                rclpy.spin_once(self, timeout_sec=0.0)         # let the camera callback fire
                cum += reward
                S.append(state); H0.append(h0); H1.append(h1); A.append(a); CR.append(cum)
                F.append(self.frame if self.frame is not None else np.zeros((600, 600, 3), np.uint8))
                ap = copy.deepcopy(a.tolist()); state = copy.deepcopy(ns)
                time.sleep(self.model.step_time)
            util.pause_simulation(self, 0)
            print(f"[{ep+1}] {util.translate_outcome(outcome)} steps={len(S)} cumreward={cum:.0f}", flush=True)
            if outcome == SUCCESS and len(S) >= 75:
                np.savez_compressed(out, states=np.array(S), h0=np.array(H0), h1=np.array(H1),
                                    actions=np.array(A), cumreward=np.array(CR), frames=np.array(F, np.uint8))
                print(f"SAVED -> {out}  ({len(S)} steps)", flush=True)
                return
        print("no suitable success episode captured", flush=True)


def main():
    rclpy.init(); Capture(sys.argv[1], sys.argv[2]); rclpy.shutdown()


if __name__ == '__main__':
    main()
