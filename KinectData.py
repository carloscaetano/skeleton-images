import math
from typing import List


class KinectJoint(object):
    def __init__(self) -> None:
        self.x_joint = 0.0
        self.y_joint = 0.0
        self.z_joint = 0.0

    def __init__(self, x_joint: float, y_joint: float, z_joint: float) -> None:
        self.x_joint = x_joint
        self.y_joint = y_joint
        self.z_joint = z_joint

class BodyData(object):
    def __init__(self) -> None:
        self.higher_x = -math.inf
        self.lower_x = math.inf
        self.higher_y = -math.inf
        self.lower_y = math.inf
        self.higher_z = -math.inf
        self.lower_z = math.inf

    def compute_higher_lower_values(self, x_joint: float, y_joint: float, z_joint: float,) -> None:
        if x_joint > self.higher_x:
            self.higher_x = x_joint
        if x_joint < self.lower_x:
            self.lower_x = x_joint
        if y_joint > self.higher_y:
            self.higher_y = y_joint
        if y_joint < self.lower_y:
            self.lower_y = y_joint
        if z_joint > self.higher_z:
            self.higher_z = z_joint
        if z_joint < self.lower_z:
            self.lower_z = z_joint

class KinectBody(object):
    def __init__(self) -> None:
        self.body_id = ''
        self.joint_data = []

    def __init__(self, body_id: str, joint_data: List[KinectJoint]) -> None:
        self.body_id = body_id
        self.joint_data = joint_data

    def __del__(self) -> None:
        del self.joint_data


class KinectBlock(object):
    def __init__(self, n_bodies: int, n_joints: int, body_list: List[KinectBody]) -> None:
        self.n_bodies = n_bodies
        self.n_joints = n_joints
        self.body_list = body_list


class KinectData(object):
    kinect_blocks: List[KinectBlock]
    perturbation_percent = 0.05

    def __init__(self) -> None:
        self.n_frames = 0
        self.n_joints = 0
        self.n_bodies = 0
        self.kinect_blocks = []
        self.n_bodies = 0
        self.body_data = []

    def __del__(self) -> None:
        del self.kinect_blocks

    def check_n_bodies(self, n_bodies: int) -> None:
        if self.n_bodies < n_bodies:
            for _ in range(n_bodies - self.n_bodies):
                self.body_data.append(BodyData())
            self.n_bodies = len(self.body_data)

    def read_block_NTU(self, file) -> None:
        """Read NTU block of Kinect data."""
        n_bodies = int(file.readline())
        self.check_n_bodies(n_bodies)
        body_list = []
        for i_body in range(n_bodies):
            data = file.readline()
            split_str = data.split(' ')
            body_id = split_str[0]
            n_joints = int(file.readline())
            joint_data = []
            for i_joint in range(n_joints):
                str_split = file.readline().split(' ')
                x_joint = float(str_split[0])
                y_joint = float(str_split[1])
                z_joint = float(str_split[2])
                joint_data.append(KinectJoint(x_joint, y_joint, z_joint))
                self.body_data[i_body].compute_higher_lower_values(x_joint, y_joint, z_joint)
            kinect_body = KinectBody(body_id, joint_data)
            body_list.append(kinect_body)
        if n_bodies > 0:
            kb = KinectBlock(n_bodies, n_joints, body_list)
        else:
            n_joints = 25
            kb = KinectBlock(n_bodies, n_joints, body_list)
        self.kinect_blocks.append(kb)
        if kb.n_bodies > self.n_bodies:
            self.n_bodies = kb.n_bodies

    def read_data(self, skl_file: str) -> None:
        """Read the Kinect data from NTU skeleton file."""
        file = open(skl_file, 'r')
        n_frames = int(file.readline())
        for _ in range(n_frames):
            self.read_block_NTU(file)
        self.n_joints = self.kinect_blocks[0].n_joints  # NTU = 25
        self.n_frames = len(self.kinect_blocks)  # Get by blocks, because there are some frames without skeleton data
        file.close()