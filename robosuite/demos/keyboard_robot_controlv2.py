#!/usr/bin/env python3
"""
python robosuite/robosuite/demos/keyboard_robot_controlv2.py --environment MoveTask --robots Panda --cameras lbview rbview --show-pov-preview --preview-cameras lbview rbview --preview-scale 1.5
Keyboard Robot Control Script with TinyVLA Data Collection

This script allows you to control a robot arm using your keyboard and collect
demonstration data in the TinyVLA H5PY format (matching playsb3_simplified_tvla_h5py_delta.py).

Data Format (matching rlds_to_h5py.py / playsb3_simplified_tvla_h5py_delta.py):
- qpos: 7D (ee_pos[3] + ee_orientation[3] + gripper[1])
- qvel: 7D (zeros)
- action: 7D - RAW delta actions (pos_delta[3] + rot_delta[3] + gripper[1])
- images: dict with camera names as keys
- language instruction stored as attribute

KEYBOARD CONTROLS:
    Z               - Discard current episode & restart (episode number unchanged)
    Ctrl+Q          - End episode (saves if successful)
    Spacebar        - Open gripper
    N               - Close gripper
    Arrow Keys      - Move horizontally in x-y plane
    . (period)      - Move up (positive z)
    ; (semicolon)   - Move down (negative z)
    O / P           - Rotate yaw
    Y / H           - Rotate pitch
    E / R           - Rotate roll
    S               - Switch active arm (multi-armed robots)
    =               - Switch active robot (multi-robot environments)

USAGE EXAMPLES:
    
    # Basic data collection:
    python keyboard_robot_control_collect.py --environment Lift --robots Panda
    
    # Different robot and task:
    python keyboard_robot_control_collect.py --environment PickPlaceCan --robots Sawyer
    
    # Custom save directory and language instruction:
    python keyboard_robot_control_collect.py --environment Lift --save-dir ./my_demos --language "Pick up the cube"
    
    # Adjust sensitivity:
    python keyboard_robot_control_collect.py --environment Lift --pos-sensitivity 2.0 --rot-sensitivity 1.5

"""

import argparse
import os

os.environ['MUJOCO_GL'] = 'glfw'

import h5py
import numpy as np
import torch
try:
    import cv2
except ImportError:
    cv2 = None

import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper
from robosuite.devices import Keyboard


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    # Camera settings - use standard robosuite camera names
    # Options: "agentview", "frontview", "sideview", "birdview", "robot0_eye_in_hand"
    "camera_names": ["lbview", "rbview"],  # Camera names for TinyVLA
    "cam_height": 180,
    "cam_width": 320,
    
    # TinyVLA dimensions (from rlds_to_h5py.py cfg)
    "state_dim": 7,   # qpos dimension: ee_pos(3) + ee_euler(3) + gripper(1)
    "action_dim": 7,  # action dimension: pos_delta(3) + rot_delta(3) + gripper(1)
    
    # Language instruction
    "language_instruction": "Move the cube to the red target area",
}


# =============================================================================
# ROTATION UTILITIES (from playsb3_simplified_tvla_h5py_delta.py)
# =============================================================================

def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.
    """
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)
    Returns:
        6D rotation representation, of size (*, 6)
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def quat_to_rot_6d(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to 6D rotation representation.
    Robosuite uses (x, y, z, w) quaternion format.
    
    Args:
        quat: Quaternion array of shape (4,)
    Returns:
        6D rotation representation of shape (6,)
    """
    rot_mat = T.quat2mat(quat)
    rot_mat_tensor = torch.from_numpy(rot_mat).float()
    rot_6d = matrix_to_rotation_6d(rot_mat_tensor)
    return rot_6d.numpy()


def euler_to_rot_6d(euler_angles: np.ndarray, convention: str = "XYZ") -> np.ndarray:
    """
    Convert Euler angles to 6D rotation representation.
    
    Args:
        euler_angles: Euler angles of shape (3,) in radians
        convention: Euler angle convention (default "XYZ")
    Returns:
        6D rotation representation of shape (6,)
    """
    euler_tensor = torch.from_numpy(euler_angles).float()
    if euler_tensor.dim() == 1:
        euler_tensor = euler_tensor.unsqueeze(0)
    rot_mat = euler_angles_to_matrix(euler_tensor, convention=convention)
    rot_6d = matrix_to_rotation_6d(rot_mat)
    return rot_6d.squeeze(0).numpy()


# =============================================================================
# STATE EXTRACTION FUNCTIONS (from playsb3_simplified_tvla_h5py_delta.py)
# =============================================================================

def get_vla_state(obs: dict) -> dict:
    """
    Extract end-effector state from robosuite observation.
    
    Returns dictionary with:
        - pos: EE position (3,)
        - rot_6d: 6D rotation representation (6,)
        - gripper: Normalized gripper state (1,)
        - quat: Original quaternion for reference (4,)
        - euler: Euler angles for qpos (3,)
    """
    # 1. EE Position (3D)
    ee_pos = obs['robot0_eef_pos'].copy()
    
    # 2. EE Rotation
    ee_quat = obs['robot0_eef_quat'].copy()
    ee_rot_6d = quat_to_rot_6d(ee_quat)
    
    # Convert quat to euler for qpos (using axis-angle as proxy)
    ee_euler = T.quat2axisangle(ee_quat)
    
    # 3. Gripper state (normalized)
    gripper_qpos = obs['robot0_gripper_qpos']
    current_width = np.sum(gripper_qpos)
    max_width = 0.08  # Panda gripper max width
    gripper_norm = np.array([np.clip(current_width / max_width, 0.0, 1.0)])
    
    return {
        "pos": ee_pos,           # (3,)
        "rot_6d": ee_rot_6d,     # (6,)
        "gripper": gripper_norm, # (1,)
        "quat": ee_quat,         # (4,)
        "euler": ee_euler,       # (3,)
    }


def state_to_qpos(vla_state: dict) -> np.ndarray:
    """
    Convert VLA state to qpos format for TinyVLA.
    
    Format: pos(3) + euler(3) + gripper(1) = 7D
    Based on rlds_to_h5py.py: traj_qpos = concat(obs_cartesian_position, obs_gripper_position)
    """
    qpos = np.concatenate([
        vla_state["pos"],      # (3,)
        vla_state["euler"],    # (3,)
        vla_state["gripper"],  # (1,)
    ])
    return qpos.astype(np.float32)


def process_image(img_raw: np.ndarray) -> np.ndarray:
    """
    Process raw image from robosuite.
    - Flip vertically (robosuite renders upside-down)
    - Ensure uint8 format
    """
    return np.flipud(img_raw).astype(np.uint8)


def update_pov_preview(obs: dict, cfg: dict):
    """
    Show a live preview window composed from one or more camera observations.
    """
    if not cfg.get("show_pov_preview", False) or cv2 is None:
        return

    preview_cams = cfg.get("preview_camera_names", [])
    if len(preview_cams) == 0:
        return

    frames = []
    for cam in preview_cams:
        img_key = cam + "_image"
        if img_key not in obs:
            continue
        frame = process_image(obs[img_key])
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(
            frame,
            cam,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        frames.append(frame)

    if len(frames) == 0:
        return

    if len(frames) == 1:
        canvas = frames[0]
    else:
        canvas = np.hstack(frames)

    preview_scale = float(cfg.get("preview_scale", 1.0))
    if preview_scale != 1.0 and preview_scale > 0:
        target_w = max(1, int(canvas.shape[1] * preview_scale))
        target_h = max(1, int(canvas.shape[0] * preview_scale))
        canvas = cv2.resize(canvas, (target_w, target_h), interpolation=cv2.INTER_AREA)

    cv2.imshow(cfg.get("preview_window_name", "POV Preview"), canvas)
    cv2.waitKey(1)


# =============================================================================
# H5PY GENERATION (from playsb3_simplified_tvla_h5py_delta.py)
# =============================================================================

def generate_h5(obs_replay: dict, action_replay: np.ndarray, cfg: dict, 
                episode_idx: int, save_dir: str, edit_flag: bool = False):
    """
    Generate HDF5 file for a single episode in TinyVLA format.
    Matches the format from rlds_to_h5py_updated.py / playsb3_simplified_tvla_h5py_delta.py.
    
    Args:
        obs_replay: Dictionary containing:
            - 'qpos': (T, 7) array of states
            - 'qvel': (T, 7) array of velocities (zeros)
            - 'images': dict of {camera_name: (T, H, W, 3) arrays}
        action_replay: (T, 7) array of actions
        cfg: Configuration dictionary
        episode_idx: Episode number for filename
        save_dir: Directory to save the file
        edit_flag: Whether data was edited (for metadata)
    
    Returns:
        Path to the saved HDF5 file
    """
    # Build data dictionary
    data_dict = {
        '/observations/qpos': obs_replay['qpos'],
        '/observations/qvel': obs_replay['qvel'],
        '/action': action_replay,
        'is_edited': np.array(edit_flag),
    }
    
    # Add images for each camera
    for cam_name in cfg['camera_names']:
        data_dict[f'/observations/images/{cam_name}'] = obs_replay['images'][cam_name]
    
    max_timesteps = len(data_dict['/observations/qpos'])
    
    # Create HDF5 file
    dataset_path = os.path.join(save_dir, f'episode_{episode_idx}.hdf5')
    
    with h5py.File(dataset_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        # Metadata attributes (matching rlds_to_h5py_updated.py)
        root.attrs['sim'] = True
        
        # Create observations group
        obs_group = root.create_group('observations')
        
        # Create images group and store camera images
        images_group = obs_group.create_group('images')
        for cam_name in cfg['camera_names']:
            images_group.create_dataset(
                cam_name, 
                (max_timesteps, cfg['cam_height'], cfg['cam_width'], 3),
                dtype='uint8',
                chunks=(1, cfg['cam_height'], cfg['cam_width'], 3),
            )
        
        # Create qpos and qvel datasets with explicit shape
        obs_group.create_dataset('qpos', (max_timesteps, cfg['state_dim']))
        obs_group.create_dataset('qvel', (max_timesteps, cfg['state_dim']))
        
        # Create action dataset with explicit shape
        root.create_dataset('action', (max_timesteps, cfg['action_dim']))
        
        # Create is_edited dataset (matching rlds_to_h5py_updated.py format)
        root.create_dataset('is_edited', (1,))
        
        # Create language_raw dataset (matching rlds_to_h5py_updated.py format)
        raw_lang = cfg['language_instruction']
        root.create_dataset("language_raw", data=[raw_lang])
        
        # Write all data
        for name, array in data_dict.items():
            root[name][...] = array
    
    return dataset_path


# =============================================================================
# TELEOPERATION WITH DATA COLLECTION
# =============================================================================

def teleoperate_and_collect(env, device, cfg: dict, arm: str = "right"):
    """
    Main control loop for teleoperating the robot with keyboard while collecting data.
    
    Args:
        env: The robosuite environment (must have camera observations enabled)
        device: The keyboard device
        cfg: Configuration dictionary with camera_names, etc.
        arm (str): Which arm to control for multi-arm robots
    
    Returns:
        Dictionary containing episode data or None if episode was cancelled/invalid
    """
    
    print("\n" + "="*70)
    print("KEYBOARD ROBOT CONTROL - TinyVLA Data Collection")
    print("="*70)
    print("Collecting data in TinyVLA H5PY format")
    print("="*70 + "\n")
    
    # Reset the discard flag at the start of each episode
    discard_flag = cfg.get("_discard_flag")
    if discard_flag is not None:
        discard_flag[0] = False
    
    obs = env.reset()
    env.render()
    update_pov_preview(obs, cfg)
    
    task_completion_hold_count = -1
    step_count = 0
    episode_success = False
    
    print("\nStarting control loop...")
    print("  Ctrl+Q = end & save  |  Z = discard & restart")
    print("Complete the task to save the demonstration!\n")
    
    # Get the action keys from the device (delta actions)
    device.start_control()
    action_dict = device.input2action()
    
    # Find the delta action key
    action_key = [key for key in action_dict.keys() if "delta" in key]
    assert len(action_key) == 1, f"Error: Expected 1 key with 'delta', found {len(action_key)}!"
    action_key = action_key[0]
    
    # Find the gripper action key
    gripper_key = [key for key in action_dict.keys() if "gripper" in key]
    assert len(gripper_key) == 1, f"Error: Expected 1 key with 'gripper', found {len(gripper_key)}!"
    gripper_key = gripper_key[0]
    
    # Storage for episode data
    all_vla_states = []
    all_images = {cam: [] for cam in cfg["camera_names"]}
    all_raw_actions = []  # Store raw delta actions (7D: pos_delta[3] + rot_delta[3] + gripper[1])
    
    # Capture initial state
    initial_vla_state = get_vla_state(obs)
    
    # Gripper settling: when toggled, allow multiple steps for it to fully actuate
    GRIPPER_SETTLE_STEPS = 3  # Number of extra steps to let gripper open/close
    gripper_steps_remaining = 0
    prev_gripper_val = None
    
    # Helper to record one step of data and advance the environment
    def _record_and_step(obs, delta_action, gripper_val, gripper_action):
        nonlocal step_count
        
        # Capture current state
        current_vla_state = get_vla_state(obs)
        all_vla_states.append(current_vla_state)
        
        # Capture images from all cameras
        for cam in cfg["camera_names"]:
            img_key = cam + "_image"
            if img_key in obs:
                img_raw = obs[img_key]
                img_processed = process_image(img_raw)
                all_images[cam].append(img_processed)
            else:
                print(f"Warning: Camera '{cam}' not in observation. Available: {[k for k in obs.keys() if 'image' in k]}")
                all_images[cam].append(np.zeros((cfg["cam_height"], cfg["cam_width"], 3), dtype=np.uint8))
        
        # Create raw 7D action for TinyVLA: pos_delta[3] + rot_delta[3] + gripper[1]
        raw_action_7d = np.concatenate([
            delta_action,
            np.array([gripper_val])
        ]).astype(np.float32)
        all_raw_actions.append(raw_action_7d)
        
        # Create full action for environment (may need padding for multi-arm)
        env_action = np.concatenate([delta_action, gripper_action if hasattr(gripper_action, '__len__') else [gripper_action]])
        
        # Fill out the rest of the action space if necessary (multi-arm robots)
        rem_action_dim = env.action_dim - env_action.size
        if rem_action_dim > 0:
            rem_action = np.zeros(rem_action_dim)
            if arm == "right":
                env_action = np.concatenate([env_action, rem_action])
            elif arm == "left":
                env_action = np.concatenate([rem_action, env_action])
        elif rem_action_dim < 0:
            env_action = env_action[:env.action_dim]
        
        # Step the environment
        new_obs, reward, done, info = env.step(env_action)
        env.render()
        update_pov_preview(new_obs, cfg)
        step_count += 1
        return new_obs, reward, done, info
    
    # Helper to check task completion, returns True if episode should end
    def _check_completion():
        nonlocal task_completion_hold_count, episode_success
        
        if task_completion_hold_count == 0:
            print("\n" + "="*70)
            print("TASK COMPLETED SUCCESSFULLY!")
            print(f"Total steps: {step_count}")
            print("="*70 + "\n")
            episode_success = True
            return True
        
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1
            else:
                task_completion_hold_count = 2
                print(f"\n✓ Success detected at step {step_count}! Holding for confirmation...")
        else:
            task_completion_hold_count = -1
        
        return False
    
    # Main control loop - only step when there is a real action
    while True:
        # Get the newest action from keyboard
        action_dict = device.input2action()
        
        # Check if discard (Z) was pressed
        if discard_flag is not None and discard_flag[0]:
            print(f"\n[Z] Episode DISCARDED at step {step_count}. Restarting...")
            return {'discarded': True}
        
        # If action is None, user pressed reset (Ctrl+Q)
        if action_dict is None:
            print(f"\nEpisode ended. Total steps: {step_count}")
            break
        
        # Get delta action and gripper from the device
        delta_action = action_dict[action_key]  # 6D: pos_delta[3] + rot_delta[3]
        gripper_action = action_dict[gripper_key]  # 1D or array
        
        # Normalize gripper to single value if it's an array
        if hasattr(gripper_action, '__len__'):
            gripper_val = gripper_action[0] if len(gripper_action) > 0 else gripper_action
        else:
            gripper_val = gripper_action
        
        # DEBUG: Print action when there's movement or gripper change
        if np.any(np.abs(delta_action) > 1e-6) or (prev_gripper_val is not None and gripper_val != prev_gripper_val):
            print(f"DEBUG delta_action: pos=[{delta_action[0]:.4f}, {delta_action[1]:.4f}, {delta_action[2]:.4f}] rot=[{delta_action[3]:.4f}, {delta_action[4]:.4f}, {delta_action[5]:.4f}] gripper={gripper_val:.4f}")
        
        # Detect gripper change → queue settling steps
        if prev_gripper_val is not None and gripper_val != prev_gripper_val:
            gripper_steps_remaining = GRIPPER_SETTLE_STEPS
            print(f"  Gripper toggled → running {GRIPPER_SETTLE_STEPS} settling steps...")
        prev_gripper_val = gripper_val
        
        # Check if there is any real action to take
        has_movement = np.any(np.abs(delta_action) > 1e-6)
        has_gripper_settling = gripper_steps_remaining > 0
        
        if not has_movement and not has_gripper_settling:
            # No action being taken — just render, don't step or record
            env.render()
            update_pov_preview(obs, cfg)
            continue
        
        if has_gripper_settling:
            # Gripper is settling: run multiple steps with zero movement delta
            zero_delta = np.zeros_like(delta_action)
            while gripper_steps_remaining > 0:
                gripper_steps_remaining -= 1
                obs, reward, done, info = _record_and_step(obs, zero_delta, gripper_val, gripper_action)
                if _check_completion():
                    break
            
            # If movement key was also held during the toggle, do that step too
            if has_movement and not episode_success:
                obs, reward, done, info = _record_and_step(obs, delta_action, gripper_val, gripper_action)
                if _check_completion():
                    break
        else:
            # Normal movement step (1 action = 1 step)
            obs, reward, done, info = _record_and_step(obs, delta_action, gripper_val, gripper_action)
            if _check_completion():
                break
        
        # Print progress every 50 steps
        if step_count % 50 == 0:
            print(f"Step {step_count} | Reward: {reward:.3f}")
    
    # Validate episode data
    T = len(all_raw_actions)
    if T < 2:
        print("Episode too short (< 2 steps), skipping...")
        return None
    
    # Ensure we have same number of states as actions
    # We should have T+1 states for T actions (initial state + state after each action)
    # But for TinyVLA format, we need T states for T actions
    if len(all_vla_states) > T:
        all_vla_states = all_vla_states[:T]
    
    # Build qpos array (T, 7)
    qpos_list = []
    for i in range(T):
        qpos_list.append(state_to_qpos(all_vla_states[i]))
    qpos_array = np.stack(qpos_list, axis=0)
    
    # Build qvel array (T, 7) - zeros like in playsb3_simplified_tvla_h5py_delta.py
    qvel_array = np.zeros_like(qpos_array)
    
    # Build action array (T, 7) - raw delta actions
    action_array = np.stack(all_raw_actions, axis=0)
    
    # Build image arrays
    images_dict = {}
    for cam_name, img_list in all_images.items():
        if len(img_list) > T:
            img_list = img_list[:T]
        images_dict[cam_name] = np.stack(img_list, axis=0)
    
    return {
        'qpos': qpos_array,
        'qvel': qvel_array,
        'action': action_array,
        'images': images_dict,
        'success': episode_success,
        'steps': step_count,
    }


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Keyboard control for robot arm with TinyVLA data collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Environment arguments
    parser.add_argument("--environment", type=str, default="Lift",
                       help="Environment/task name (e.g., Lift, PickPlaceCan, Stack, NutAssembly)")
    parser.add_argument("--robots", nargs="+", type=str, default=["Panda"],
                       help="Robot(s) to use (e.g., Panda, Sawyer, UR5e, Baxter)")
    parser.add_argument("--config", type=str, default="default",
                       help="Environment configuration (for TwoArm environments: bimanual, parallel, opposed)")
    parser.add_argument("--arm", type=str, default="right",
                       help="Which arm to control for multi-arm robots (right/left)")
    
    # Controller arguments
    parser.add_argument("--controller", type=str, default=None,
                       help="Controller type (e.g., BASIC, OSC_POSE, WHOLE_BODY_MINK_IK)")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0,
                       help="Position input sensitivity multiplier")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0,
                       help="Rotation input sensitivity multiplier")
    
    # Data collection arguments
    parser.add_argument("--save-dir", type=str, default=None,
                       help="Directory to save demonstrations (default: ./demonstrations_tinyvla/)")
    parser.add_argument("--language", type=str, default="Move the cube to the target red area",
                       help="Language instruction for the task")
    parser.add_argument("--only-success", action="store_true",
                       help="Only save successful demonstrations")
    
    # Camera arguments
    # Standard robosuite cameras: agentview, frontview, sideview, birdview, robot0_eye_in_hand
    parser.add_argument("--cameras", nargs="+", type=str, default=["agentview", "sideview"],
                       help="Camera names for data collection (e.g., agentview, frontview, sideview)")
    parser.add_argument("--cam-height", type=int, default=180,
                       help="Camera image height")
    parser.add_argument("--cam-width", type=int, default=320,
                       help="Camera image width")
    parser.add_argument("--show-pov-preview", action="store_true",
                       help="Show live side-by-side preview window from offscreen camera feeds")
    parser.add_argument("--preview-cameras", nargs="+", type=str, default=None,
                       help="Camera names for preview window (defaults to first two cameras from --cameras)")
    parser.add_argument("--preview-scale", type=float, default=1.0,
                       help="Scale factor for preview window (e.g., 1.5 for larger preview)")
    
    # Control arguments
    parser.add_argument("--control-freq", type=int, default=20,
                       help="Control frequency in Hz")
    
    args = parser.parse_args()
    
    # Set up configuration
    cfg = {
        "camera_names": args.cameras,
        "cam_height": args.cam_height,
        "cam_width": args.cam_width,
        "state_dim": DEFAULT_CONFIG["state_dim"],
        "action_dim": DEFAULT_CONFIG["action_dim"],
        "language_instruction": args.language,
        "show_pov_preview": args.show_pov_preview,
        "preview_scale": args.preview_scale,
        "preview_window_name": "TinyVLA POV Preview",
    }

    preview_camera_names = args.preview_cameras if args.preview_cameras is not None else cfg["camera_names"][:2]
    cfg["preview_camera_names"] = preview_camera_names

    if cfg["show_pov_preview"] and cv2 is None:
        print("Warning: --show-pov-preview requested, but OpenCV is not installed. Install with: pip install opencv-python")
        cfg["show_pov_preview"] = False

    # Ensure preview cameras are available in observations, without changing saved data cameras
    env_camera_names = list(dict.fromkeys(cfg["camera_names"] + cfg["preview_camera_names"]))
    
    # Set up save directory
    if args.save_dir is None:
        args.save_dir = os.path.join(os.getcwd(), "demonstrations_tinyvla")
    os.makedirs(args.save_dir, exist_ok=True)

    # Use a single stable directory per environment and keep appending episodes there
    env_folder = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in args.environment).strip("_")
    if not env_folder:
        env_folder = "environment"
    env_dir = os.path.join(args.save_dir, env_folder)
    os.makedirs(env_dir, exist_ok=True)

    print(f"\nEnvironment directory: {env_dir}")
    
    # Load controller configuration
    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots[0],
    )
    
    # Build environment configuration
    env_config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }
    
    # Handle multi-arm environments
    if "TwoArm" in args.environment:
        env_config["env_configuration"] = args.config
    
    # Create environment with camera observations
    print("\nCreating environment with camera observations...")
    
    env = suite.make(
        **env_config,
        has_renderer=True,
        has_offscreen_renderer=True,  # Enable offscreen rendering for camera obs
        render_camera="frontview",
        ignore_done=True,
        use_camera_obs=True,  # Enable camera observations
        camera_names=env_camera_names,
        camera_heights=[cfg["cam_height"]] * len(env_camera_names),
        camera_widths=[cfg["cam_width"]] * len(env_camera_names),
        reward_shaping=True,
        control_freq=args.control_freq,
        hard_reset=False,
    )
    
    # Add visualization wrapper
    env = VisualizationWrapper(env, indicator_configs=None)
    
    # Hide gripper visualization (the green marker through the gripper)
    env.set_visualization_setting("grippers", False)
    
    # Initialize keyboard device
    print("Initializing keyboard controller...")
    device = Keyboard(
        env=env, 
        pos_sensitivity=args.pos_sensitivity, 
        rot_sensitivity=args.rot_sensitivity
    )
    
    # Set up discard flag and monkey-patch device.on_press to intercept Z key
    discard_flag = [False]
    cfg["_discard_flag"] = discard_flag
    _original_on_press = device.on_press

    def _patched_on_press(key):
        try:
            if hasattr(key, 'char') and key.char == 'z':
                discard_flag[0] = True
                return
        except Exception:
            pass
        _original_on_press(key)

    device.on_press = _patched_on_press

    # Register keyboard callback with the viewer (required for mjviewer)
    env.viewer.add_keypress_callback(_patched_on_press)
    
    # Set up numpy printing
    np.set_printoptions(formatter={"float": lambda x: f"{x:0.3f}"})
    
    # Find existing episodes in environment directory
    existing_episodes = set()
    for fname in os.listdir(env_dir):
        if fname.startswith("episode_") and fname.endswith(".hdf5"):
            try:
                ep_num = int(fname.split('_')[1].split('.')[0])
                existing_episodes.add(ep_num)
            except (IndexError, ValueError):
                continue
    
    # Main loop - collect multiple demonstrations
    demo_count = 0
    saved_count = 0
    ep_idx = max(existing_episodes, default=-1) + 1
    
    try:
        if cfg["show_pov_preview"]:
            print(f"Live POV preview enabled: {cfg['preview_camera_names']}")
        while True:
            demo_count += 1
            print(f"\n{'='*70}")
            print(f"DEMONSTRATION #{demo_count} (Episode index: {ep_idx})")
            print(f"{'='*70}\n")
            
            # Run teleoperation and collect data
            episode_data = teleoperate_and_collect(
                env, 
                device, 
                cfg,
                arm=args.arm,
            )
            
            # Check if episode should be saved
            if episode_data is not None and episode_data.get('discarded'):
                print("Episode discarded by user (Z key). Restarting...")
                continue
            elif episode_data is None:
                print("Episode invalid, not saved.")
            elif args.only_success and not episode_data['success']:
                print("Episode unsuccessful and --only-success is set, not saved.")
            else:
                # Prepare observation replay dict
                obs_replay = {
                    'qpos': episode_data['qpos'],
                    'qvel': episode_data['qvel'],
                    'images': episode_data['images'],
                }
                
                # Generate H5PY file
                h5_path = generate_h5(
                    obs_replay=obs_replay,
                    action_replay=episode_data['action'],
                    cfg=cfg,
                    episode_idx=ep_idx,
                    save_dir=env_dir,
                    edit_flag=False,
                )
                
                saved_count += 1
                ep_idx += 1
                
                print(f"\n--- Episode Saved ---")
                print(f"qpos shape: {episode_data['qpos'].shape}")
                print(f"qvel shape: {episode_data['qvel'].shape}")
                print(f"action shape: {episode_data['action'].shape}")
                for cam, imgs in episode_data['images'].items():
                    print(f"images[{cam}] shape: {imgs.shape}")
                print(f"Success: {episode_data['success']}")
                print(f"Saved to: {h5_path}")
                print(f"---------------------\n")
            
            # Ask if user wants to continue
            print("\nOptions:")
            print("  Press ENTER to collect another demonstration")
            print("  Type 'q' and press ENTER to quit")
            user_input = input("\nYour choice: ").strip().lower()
            
            if user_input == 'q':
                break
                
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        if cfg["show_pov_preview"] and cv2 is not None:
            cv2.destroyAllWindows()
        # Clean up environment when done
        env.close()
    
    print("\n" + "="*70)
    print("SESSION COMPLETE")
    print(f"Total demonstrations attempted: {demo_count}")
    print(f"Total demonstrations saved: {saved_count}")
    print(f"Data saved to: {env_dir}")
    print("="*70 + "\n")


def verify_h5_file(filepath: str):
    """Utility function to verify an H5PY file structure."""
    print(f"\nVerifying: {filepath}")
    with h5py.File(filepath, 'r') as f:
        print("\nAttributes:")
        for key, val in f.attrs.items():
            print(f"  {key}: {val}")
        
        print("\nDatasets:")
        def print_structure(group, indent=0):
            for name in group:
                item = group[name]
                prefix = "  " * indent
                if isinstance(item, h5py.Group):
                    print(f"{prefix}{name}/")
                    print_structure(item, indent + 1)
                elif isinstance(item, h5py.Dataset):
                    print(f"{prefix}{name}: shape={item.shape}, dtype={item.dtype}")
        
        print_structure(f)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        # Verify mode: check an existing H5 file
        if len(sys.argv) > 2:
            verify_h5_file(sys.argv[2])
        else:
            print("Usage: python keyboard_robot_control_collect.py --verify <path_to_h5_file>")
    else:
        # Normal collection mode
        main()
