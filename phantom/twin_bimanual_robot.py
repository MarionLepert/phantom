"""
Virtual twin bimanual robot implementation for MuJoCo simulation.

This module provides a TwinBimanualRobot class that creates a virtual representation
of a bimanual (two-arm) robot system in MuJoCo using the robosuite framework.
The twin robot can be controlled via end-effector poses or joint positions and
provides observation data including RGB images, depth maps, and robot masks.
"""

from collections import deque
import re
import cv2
import pdb
import matplotlib.pyplot as plt
import numpy as np 
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
from typing import Tuple, Union, Any

from robosuite.controllers import load_controller_config # type: ignore
from robosuite.utils.camera_utils import get_real_depth_map # type: ignore
from robomimic.envs.env_robosuite import EnvRobosuite # type: ignore
import robomimic.utils.obs_utils as ObsUtils # type: ignore


@dataclass
class MujocoCameraParams:
    """
    Camera parameters for MuJoCo simulation.
    
    Attributes:
        name: Camera name identifier
        pos: 3D position of camera in world coordinates
        ori_wxyz: Camera orientation as quaternion (w, x, y, z)
        fov: Field of view in degrees
        resolution: Image resolution as (width, height)
        sensorsize: Physical sensor size in mm
        principalpixel: Principal point coordinates in pixels
        focalpixel: Focal length in pixels
    """
    name: str
    pos: np.ndarray
    ori_wxyz: np.ndarray
    fov: float
    resolution: Tuple[int, int]
    sensorsize: np.ndarray
    principalpixel: np.ndarray
    focalpixel: np.ndarray

# Color constants for visualization (RGBA format)
THUMB_COLOR = [0, 1, 0, 1]  # Green for thumb
INDEX_COLOR = [1, 0, 0, 1]  # Red for index finger
HAND_EE_COLOR = [0, 0, 1, 1]  # Blue for hand end-effector

# Transformation matrix for Epic Kitchen setup - converts from base frame to robot frame
BASE_T_1 = np.array([[0.0, -1.0,  0.0,  0.0],
                    [ 0.5,  0.0,  0.866,  0.2],
                    [-0.866,  0.0,  0.5,  1.50],
                    [ 0.0,  0.0,  0.0,  1.0]])

# Bimanual layout for ARX R5 arms: shoulder-mounted arms 0.24 m apart, each tilted ~60 deg outwards.
# Base poses in the layout frame (x right, y forward, z up) as (xyz, URDF rpy). For ARXR5 this layout frame is
# phantom's robot frame "1", so the camera extrinsics json must give the camera pose in this frame.
ARX_R5_BASE_POSES = {
    "right": ([0.12, 0.0, -0.04], [1.05, 0.0, np.pi / 2]),   # robot0
    "left": ([-0.12, 0.0, -0.04], [-1.05, 0.0, np.pi / 2]),  # robot1
}

# Layout frame -> MuJoCo world: lift the arms above the empty arena floor
ARX_R5_T_1 = np.array([[1.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 1.5],
                       [0.0, 0.0, 0.0, 1.0]])

def convert_real_camera_ori_to_mujoco(camera_ori_matrix: np.ndarray) -> np.ndarray:
    """
    Convert camera orientation from real world to MuJoCo XML format.
    
    MuJoCo uses a different coordinate system convention, so we need to
    flip the Y and Z axes of the rotation matrix before converting to quaternion.
    
    Args:
        camera_ori_matrix: 3x3 rotation matrix in real-world coordinates
        
    Returns:
        Camera orientation as quaternion in MuJoCo format (w, x, y, z)
    """
    camera_ori_matrix[:, [1, 2]] = -camera_ori_matrix[:, [1, 2]]
    r = Rotation.from_matrix(camera_ori_matrix)
    camera_ori_wxyz = r.as_quat(scalar_first=True)
    return camera_ori_wxyz

class TwinBimanualRobot:
    """
    Virtual twin of a bimanual robot system in MuJoCo simulation.
    
    This class creates a simulated bimanual robot that can be controlled via
    end-effector poses or joint positions. It provides functionality for:
    - Robot pose control (OSC or joint-level)
    - Camera observation collection (RGB, depth, segmentation)
    - Robot and gripper mask generation
    - Observation history management
    """
    
    def __init__(self, robot_name: str, gripper_name: str, bimanual_setup: str,
                 camera_params: MujocoCameraParams, camera_height: int, camera_width: int,
                 render: bool, n_steps_short: int, n_steps_long: int, square: bool = False,
                 debug_cameras: list[str] = [], epic: bool = False, joint_controller: bool = False): 
        """
        Initialize the bimanual robot twin.
        
        Args:
            robot_name: Type of robot (e.g., "Kinova3")
            gripper_name: Type of gripper (e.g., "Robotiq85")
            bimanual_setup: Configuration for bimanual setup
            camera_params: Camera configuration parameters
            camera_height: Height of camera images in pixels
            camera_width: Width of camera images in pixels
            render: Whether to render the simulation visually
            n_steps_short: Number of simulation steps for quick movements
            n_steps_long: Number of simulation steps for initial/slow movements
            square: Whether to crop images to square aspect ratio
            debug_cameras: Additional camera names for debugging views
            epic: Whether to use Epic Kitchen coordinate system
            joint_controller: Whether to use joint-level control instead of OSC
        """
        # Store configuration parameters
        self.robot_name = robot_name
        self.gripper_name = gripper_name
        self.bimanual_setup = bimanual_setup
        self.camera_params = camera_params
        self.render = render
        self.n_steps_long = n_steps_long
        self.n_steps_short= n_steps_short
        self.num_frames = 2  # Number of frames to keep in observation history
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.camera_name = "zed"  # Main camera name
        self.square = square
        self.debug_cameras = list(debug_cameras) if debug_cameras else []
        self.epic = epic  # Epic Kitchen mode flag
        self.joint_controller = joint_controller  # Control mode flag

        # Configure observation specifications for robomimic
        obs_spec = dict(
            obs=dict(
                low_dim=["robot0_eef_pos"],  # End-effector position observations
                rgb=[f"{self.camera_params.name}_image"] + [f"{cam}_image" for cam in self.debug_cameras],
            ),
        )
        ObsUtils.initialize_obs_utils_with_obs_specs(
            obs_modality_specs=obs_spec)
                        
        # Configure robosuite environment options
        options: dict[str, Union[str, list[str], dict[str, Any], bool, int, np.ndarray]] = {}
        options["env_name"] = "PhantomBimanual"
        options["bimanual_setup"] = bimanual_setup
        options["robots"] = [self.robot_name, self.robot_name]  # Two identical robots
        if self.robot_name == "Kinova3":
            options["gripper_types"] = [f"{self.gripper_name}GripperRealKinova", f"{self.gripper_name}GripperRealKinova"]
        else:
            options["gripper_types"] = [f"{self.gripper_name}Gripper", f"{self.gripper_name}Gripper"]
        
        # Configure controller (OSC pose control by default)
        controller_config = load_controller_config(default_controller="OSC_POSE")
        controller_config["control_delta"] = False  # Use absolute positioning
        controller_config["uncouple_pos_ori"] = False  # Couple position and orientation
        options["controller_configs"] = controller_config
        
        # Override with joint controller if specified
        if self.joint_controller:
            controller_config = load_controller_config(default_controller="JOINT_POSITION")
            controller_config["input_type"] = "absolute"
            controller_config["input_max"] = 10
            controller_config["input_min"] = -10
            controller_config["output_max"] = 10
            controller_config["output_min"] = -10
            controller_config["kd"] = 200  # Derivative gain
            controller_config["kv"] = 200  # Velocity gain
            controller_config["kp"] = 1000  # Proportional gain
            controller_config["kp_limits"] = [0, 1000]  # Proportional gain limits
            options["controller_configs"] = controller_config
            
        # Camera and observation settings
        options["camera_heights"] = self.camera_height
        options["camera_widths"] = self.camera_width
        options["camera_segmentations"] = "instance"  # Instance segmentation masks
        options["direct_gripper_control"] = True
        options["use_depth_obs"] = True

        # ARX R5 arms use ARX_R5_BASE_POSES instead of the hardcoded Kinova shoulders layout
        if self.robot_name == "ARXR5":
            base_poses = []
            for side in ("right", "left"):
                pos, rpy = ARX_R5_BASE_POSES[side]
                world_R = ARX_R5_T_1[:3, :3] @ Rotation.from_euler("xyz", rpy).as_matrix()
                world_pos = ARX_R5_T_1[:3, :3] @ np.array(pos) + ARX_R5_T_1[:3, 3]
                base_poses.append((world_pos, Rotation.from_matrix(world_R).as_quat(scalar_first=True)))
            options["robot_base_poses"] = base_poses
        
        # Apply Epic Kitchen coordinate transformation if enabled
        if self.epic:
            self.base_T_1 = ARX_R5_T_1 if self.robot_name == "ARXR5" else BASE_T_1
            # Transform camera position and orientation to Epic Kitchen frame
            self.camera_params.pos = self.base_T_1[:3, :3] @ self.camera_params.pos + self.base_T_1[:3, 3]
            camera_ori_matrix = self.base_T_1[:3, :3] @ Rotation.from_quat(self.camera_params.ori_wxyz, scalar_first=True).as_matrix()
            self.camera_params.ori_wxyz = Rotation.from_matrix(camera_ori_matrix).as_quat(scalar_first=True)

        # Set camera parameters
        options["camera_pos"] = self.camera_params.pos
        options["camera_quat_wxyz"] = self.camera_params.ori_wxyz
        options["camera_sensorsize"] = self.camera_params.sensorsize
        options["camera_principalpixel"] = self.camera_params.principalpixel
        options["camera_focalpixel"] = self.camera_params.focalpixel

        # Create the robosuite environment
        self.env = EnvRobosuite(
            **options,
            render=render,
            render_offscreen=True,  # Enable offscreen rendering for image capture
            use_image_obs=True,
            camera_names=[self.camera_params.name] + self.debug_cameras,
            control_freq=20,  # 20 Hz control frequency
        )

        # Initialize environment and compute robot base position
        self.reset()
        self.robot_base_pos = np.array([0, 0, self.env.env.robot_base_height+self.env.env.robot_base_offset])

 
    def reset(self):
        """Reset environment and clear observation history."""
        self.env.reset()
        self.obs_history = deque()

    def close(self):
        """Close the simulation environment."""
        self.env.env.close()

    def get_action_from_ee_pose(self, ee_pos: np.ndarray, ee_quat_xyzw: np.ndarray, gripper_action: float,
                                use_base_offset: bool = False) -> np.ndarray:
        """
        Convert end-effector pose to robot action vector.
        
        This method transforms the desired end-effector position and orientation
        into the action format expected by the robot controller.
        
        Args:
            ee_pos: End-effector position as 3D array
            ee_quat_xyzw: End-effector orientation as quaternion (x, y, z, w)
            gripper_action: Gripper action value
            use_base_offset: Whether to add robot base offset to position
            
        Returns:
            Action vector [position(3), rotation(3), gripper(1)]
        """
        # Handle batch inputs by taking the last element
        if ee_pos.ndim > 1:
            ee_pos = ee_pos[-1]
            ee_quat_xyzw = ee_quat_xyzw[-1]
            
        # Add base offset if requested and not in Epic mode
        if use_base_offset and not self.epic:
            ee_pos = ee_pos + self.robot_base_pos
        
        # Apply coordinate transformations based on mode
        if self.epic:
            # Transform position and orientation to Epic Kitchen coordinate frame
            ee_pos = self.base_T_1[:3, 3] + self.base_T_1[:3, :3] @ ee_pos
            axis_angle = Rotation.from_matrix(self.base_T_1[:3, :3] @ Rotation.from_quat(ee_quat_xyzw).as_matrix()).as_rotvec()
        elif not self.epic:
            # Apply 135-degree Z rotation for standard setup
            rot = Rotation.from_quat(ee_quat_xyzw)
            rot_135deg = Rotation.from_euler('z', 135, degrees=True)
            new_rot = rot * rot_135deg 
            axis_angle = new_rot.as_rotvec()
            
        # Combine into action vector
        action = np.concatenate([ee_pos, axis_angle, [gripper_action]])

        return action

    def _get_initial_obs_history(self, state: dict) -> deque:
        """
        Initialize observation history by repeating the first observation.
        
        This creates a history buffer filled with the initial robot state,
        which is useful for algorithms that require temporal context.
        
        Args:
            state: Initial robot state dictionary
            
        Returns:
            Deque containing repeated initial observations
        """
        obs_history = deque(
                [self.move_to_target_state(state, init=True)], 
                maxlen=self.num_frames,
        )
        # Fill remaining slots with copies of the initial observation
        for _ in range(self.num_frames-1):
            obs_history.append(self.move_to_target_state(state))
        return obs_history
    
    def get_obs_history(self, state: dict) -> list:
        """
        Get observation history with specified length.
        
        Maintains a rolling buffer of recent observations for temporal context.
        
        Args:
            state: Current robot state dictionary
            
        Returns:
            List of recent observations (length = self.num_frames)
        """
        if len(self.obs_history) == 0:
            # Initialize history if empty
            self.obs_history = self._get_initial_obs_history(state)
        else:
            # Add new observation to history
            self.obs_history.append(self.move_to_target_state(state))
        return list(self.obs_history)
    
    def move_to_target_state(self, state: dict, init=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Move robot to target state and collect observation data.
        
        This is the main method for controlling the robot and collecting observations.
        It handles both pose and joint control modes, and collects RGB, depth,
        and segmentation data along with tracking errors.
        
        Args:
            state: Target state containing positions, orientations, and gripper states
            init: Whether this is an initialization step (uses longer movement time)
            
        Returns:
            Dictionary containing observation data:
            - robot_mask: Binary mask showing robot pixels
            - gripper_mask: Binary mask showing gripper pixels  
            - rgb_img: RGB camera image
            - depth_img: Depth camera image
            - robot_pos: Robot end-effector position
            - left_pos_err: Left arm position tracking error
            - right_pos_err: Right arm position tracking error
            - {cam}_img: Additional camera images if debug_cameras specified
        """
        # Convert gripper positions to actions based on controller type
        if not self.joint_controller:
            # Use pose controller with gripper position mapping
            gripper_action_0 = self._convert_handgripper_pos_to_action(state["gripper_pos"][0])
            gripper_action_1 = self._convert_handgripper_pos_to_action(state["gripper_pos"][1])
            gripper_action = [gripper_action_0, gripper_action_1]
        else:
            # Use joint controller with direct gripper control
            gripper_action = [state["gripper_pos"][0]*255, state["gripper_pos"][1]*255]

        # Choose movement duration based on whether this is initialization
        n_steps = self.n_steps_long if init else self.n_steps_short

        # Execute movement based on controller type
        if not self.joint_controller:
            # Move using pose control
            obs = self.move_to_pose(state["pos"], state["ori_xyzw"], gripper_action, n_steps)
        else:
            # Move using joint control
            obs = self.move_to_pose(state["pos"], state["ori_xyzw"], gripper_action, n_steps, state["q0"], state["q1"])

        # Extract observation data from simulation
        robot_mask = np.squeeze(self.get_robot_mask(obs))
        gripper_mask = np.squeeze(self.get_gripper_mask(obs))
        rgb_img = self.get_image(obs)
        depth_img = self.get_depth_image(obs)
        robot_pos = obs["robot0_eef_pos"] - self.robot_base_pos

        # Calculate end-effector tracking errors for both arms
        if not self.epic:
            # Standard coordinate frame
            right_pos_error = np.linalg.norm(obs['robot0_eef_pos']-self.robot_base_pos - state["pos"][0])
            left_pos_error = np.linalg.norm(obs['robot1_eef_pos']-self.robot_base_pos - state["pos"][1])
        else:
            # Epic Kitchen coordinate frame
            right_pos_error = np.linalg.norm(obs['robot0_eef_pos']-self.base_T_1[:3, 3] - self.base_T_1[:3, :3] @ state["pos"][0])
            left_pos_error = np.linalg.norm(obs['robot1_eef_pos']-self.base_T_1[:3, 3] - self.base_T_1[:3, :3] @ state["pos"][1])

        # Compile output dictionary
        output = {
            "robot_mask": robot_mask,
            "gripper_mask": gripper_mask,
            "rgb_img": rgb_img,
            "depth_img": depth_img,
            "robot_pos": robot_pos,
            "left_pos_err": left_pos_error,
            "right_pos_err": right_pos_error,
        }

        # Add debug camera images if specified
        for cam in self.debug_cameras:
            cam_img = self.get_camera_image(obs, cam)
            output[f"{cam}_img"] = cam_img

        return output
 
    def _convert_handgripper_pos_to_action(self, gripper_pos: float) -> np.ndarray:
        """
        Convert hand gripper position to robot gripper action.
        
        Maps from physical gripper opening distance to robot action values.
        Different gripper types may have different mappings.
        
        Args:
            gripper_pos: Gripper opening distance in meters
            
        Returns:
            Robot gripper action value (0-255 for Robotiq85)
            
        Raises:
            ValueError: If gripper type is not supported
        """
        if self.gripper_name == "Robotiq85":
            # Robotiq85 gripper specifications
            min_gripper_pos, max_gripper_pos = 0.0, 0.085  # 0 to 8.5cm opening
            gripper_pos = np.clip(gripper_pos, min_gripper_pos, max_gripper_pos)
            open_gripper_action, closed_gripper_action = 0, 255  # 0=open, 255=closed
            # Linear interpolation between open and closed states
            return np.interp(gripper_pos, [min_gripper_pos, max_gripper_pos], [closed_gripper_action, open_gripper_action])
        elif self.gripper_name == "ARXR5":
            # ARX R5 parallel gripper: 0 to 8.8cm total opening, two fingers with 0-4.4cm travel each.
            # Action is the per-finger travel in meters; robosuite applies [a, -a] to the two fingers.
            min_gripper_pos, max_gripper_pos = 0.0, 0.088
            gripper_pos = np.clip(gripper_pos, min_gripper_pos, max_gripper_pos)
            return gripper_pos / 2.0
        else:
            raise ValueError(f"Gripper name {self.gripper_name} not supported")

    def move_to_pose(self, ee_pos: dict, ee_ori: dict, gripper_action: dict, n_steps: int, q0=None, q1=None) -> dict:
        """
        Execute robot movement to target pose.
        
        Sends action commands to the simulation for the specified number of steps.
        Handles both pose control (OSC) and joint control modes.
        
        Args:
            ee_pos: End-effector positions for both arms {0: pos0, 1: pos1}
            ee_ori: End-effector orientations for both arms {0: ori0, 1: ori1}
            gripper_action: Gripper actions for both arms {0: grip0, 1: grip1}
            n_steps: Number of simulation steps to execute
            q0: Joint positions for arm 0 (only for joint controller)
            q1: Joint positions for arm 1 (only for joint controller)
            
        Returns:
            Final observation dictionary from simulation
        """
        if not self.joint_controller:
            # Pose control mode: convert poses to actions
            action_0 = self.get_action_from_ee_pose(ee_pos[0], ee_ori[0], gripper_action[0], use_base_offset=True)
            action_1 = self.get_action_from_ee_pose(ee_pos[1], ee_ori[1], gripper_action[1], use_base_offset=True)
            action = np.concatenate([action_0, action_1])
        else:
            # Joint control mode: convert joint angles from degrees to radians
            q0_new = []
            for rot_q in q0:
                if rot_q >= 180:
                    q0_new.append((rot_q/180*np.pi-2*np.pi))  # Handle angle wrapping
                else:
                    q0_new.append(rot_q/180*np.pi)
            q1_new = []
            for rot_q in q1:
                if rot_q >= 180:
                    q1_new.append((rot_q/180*np.pi-2*np.pi))  # Handle angle wrapping
                else:
                    q1_new.append(rot_q/180*np.pi)

            # Combine joint positions and gripper actions
            action_0 = q0_new
            action_1 = q1_new
            action = np.concatenate([action_0, np.array(gripper_action[0]).reshape(1,), action_1, np.array(gripper_action[1]).reshape(1,)])

        # Execute action for specified number of steps
        for _ in range(n_steps):
            obs, _, _, _ = self.env.step(action)
            if self.render:
                self.env.render()
        return obs
    
    def get_proprioception(self, obs: dict) -> np.ndarray:
        """
        Get proprioceptive information (robot's internal state).
        
        Args:
            obs: Observation dictionary from simulation
            
        Returns:
            End-effector position of first robot
        """
        pos = obs["robot0_eef_pos"]
        return pos

    def get_image(self, obs: dict) -> np.ndarray:
        """
        Extract RGB image from observation.
        
        Handles image format conversion and optional square cropping.
        
        Args:
            obs: Observation dictionary containing image data
            
        Returns:
            RGB image as numpy array (H, W, 3)
        """
        img = obs[f"{self.camera_name}_image"]
        img = img.transpose(1, 2, 0)  # Convert from CHW to HWC format
        height = img.shape[0]
        width = img.shape[1]
        
        # Crop to square if requested
        if self.square:
            n_remove = int((width - height)/2)
            img = img[:,n_remove:-n_remove,:]
        return img
    
    def get_camera_image(self, obs: dict, camera_name: str) -> np.ndarray:
        """
        Extract RGB image from specific camera.
        
        Args:
            obs: Observation dictionary containing image data
            camera_name: Name of the camera to extract image from
            
        Returns:
            RGB image as numpy array (H, W, 3)
        """
        img = obs[f"{camera_name}_image"]
        img = img.transpose(1, 2, 0)  # Convert from CHW to HWC format
        height = img.shape[0]
        width = img.shape[1]
        
        # Crop to square if requested
        if self.square:
            n_remove = int((width - height)/2)
            img = img[:,n_remove:-n_remove,:]
        return img
    
    def get_seg_image(self, obs: dict) -> np.ndarray:
        """
        Extract instance segmentation image.
        
        Args:
            obs: Observation dictionary containing segmentation data
            
        Returns:
            Segmentation image as uint8 array where each pixel value
            represents a different object instance ID
        """
        img = obs[f"{self.camera_name}_segmentation_instance"]
        height = img.shape[0]
        width = img.shape[1]
        
        # Crop to square if requested
        if self.square:
            n_remove = int((width - height)/2)
            img = img[:,n_remove:-n_remove,:]  
            img = img.astype(np.uint8)
        return img

    def get_depth_image(self, obs: dict) -> np.ndarray:
        """
        Extract and process depth image.
        
        Converts raw depth buffer to real-world depth values using
        robosuite's depth processing utilities.
        
        Args:
            obs: Observation dictionary containing depth data
            
        Returns:
            Depth image as numpy array where values represent
            distance in meters
        """
        img = obs[f"{self.camera_name}_depth"]
        img = get_real_depth_map(sim=self.env.env.sim, depth_map=img)
        height = img.shape[0]
        width = img.shape[1]
        
        # Crop to square if requested
        if self.square:
            n_remove = int((width - height)/2)
            img = img[:,n_remove:-n_remove,:]
        return img
    
    def get_robot_mask(self, obs: dict) -> np.ndarray:
        """
        Generate binary mask for robot pixels.
        
        Uses instance segmentation to identify which pixels belong to
        the robot arms (instance IDs 1 and 4).
        
        Args:
            obs: Observation dictionary containing segmentation data
            
        Returns:
            Binary mask where 1 indicates robot pixels, 0 otherwise
        """
        seg_img = self.get_seg_image(obs)
        mask = np.zeros_like(seg_img)
        mask[seg_img == 1] = 1  # First robot arm
        mask[seg_img == 4] = 1  # Second robot arm
        return mask
    
    def get_gripper_mask(self, obs: dict) -> np.ndarray:
        """
        Generate binary mask for gripper pixels.
        
        Uses instance segmentation to identify which pixels belong to
        the robot grippers (instance IDs 3 and 6).
        
        Args:
            obs: Observation dictionary containing segmentation data
            
        Returns:
            Binary mask where 1 indicates gripper pixels, 0 otherwise
        """
        seg_img = self.get_seg_image(obs)
        mask = np.zeros_like(seg_img)
        mask[seg_img == 3] = 1  # First gripper
        mask[seg_img == 6] = 1  # Second gripper
        return mask
