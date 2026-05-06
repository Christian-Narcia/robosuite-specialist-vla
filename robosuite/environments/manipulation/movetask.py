from collections import OrderedDict
import os

# Fix OpenMP duplicate library error on Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.transform_utils import convert_quat


class MoveTask(ManipulationEnv):
    """
    Move task environment where the gripper starts grasping a randomly spawned cube
    and must move it to a target zone.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task.

        gripper_types (str or list of str): type of gripper, used to instantiate gripper models from gripper factory.

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will be used to place objects on every reset.

        has_renderer (bool): If true, render the simulation state in a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True.

        render_collision_mesh (bool): True if rendering collision meshes in camera.

        render_visual_mesh (bool): True if rendering visual meshes in camera.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.

        control_freq (float): how many control signals to receive in every second.

        lite_physics (bool): Whether to optimize for mujoco forward and step calls.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call.

        camera_names (str or list of str): name of camera to be rendered.

        camera_heights (int or list of int): height of camera frame.

        camera_widths (int or list of int): width of camera frame.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=500,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mjviewer",
        renderer_config=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):
        """
        Reward function for move task.
        Simple distance-based reward: negative XY distance + negative Z distance.
        """
        # Get positions
        cube_pos = np.array(self.sim.data.body_xpos[self.cube_body_id])
        target_pos = np.array(self.sim.data.body_xpos[self.target_zone_body_id])
        
        # XY distance (horizontal)
        xy_dist = np.linalg.norm(cube_pos[:2] - target_pos[:2])
        
        # Z distance (vertical) - target Z is table height + cube half-height
        target_z = self.table_offset[2] + self.cube_size[2]
        z_dist = abs(cube_pos[2] - target_z)
        
        # Simple negative distance reward (closer = higher reward)
        reward = -xy_dist - z_dist
        
        # Bonus for success
        if self._check_success():
            reward += 10.0

        # Scale
        if self.reward_scale is not None:
            reward *= self.reward_scale

        return reward

    def _cube_to_target_distance(self):
        """
        Calculate the XY + Z distance between the cube and the target.

        Returns:
            tuple: (xy_dist, z_dist)
        """
        cube_pos = self.sim.data.body_xpos[self.cube_body_id]
        target_pos = self.sim.data.body_xpos[self.target_zone_body_id]
        
        xy_dist = np.linalg.norm(cube_pos[:2] - target_pos[:2])
        target_z = self.table_offset[2] + self.cube_size[2]
        z_dist = abs(cube_pos[2] - target_z)
        
        return xy_dist, z_dist

    def _cube_on_target(self):
        """
        Check if the cube is at the target zone.

        Returns:
            bool: True if cube is within threshold distance of target (XY and Z)
        """
        cube_pos = np.array(self.sim.data.body_xpos[self.cube_body_id])
        target_pos = np.array(self.sim.data.body_xpos[self.target_zone_body_id])
        
        # XY distance
        xy_dist = np.linalg.norm(cube_pos[:2] - target_pos[:2])
        
        # Z distance - target Z is table height + cube half-height
        target_z = self.table_offset[2] + self.cube_size[2]
        z_dist = abs(cube_pos[2] - target_z)
        
        # Success if close in XY and Z
        return xy_dist < 0.05 and z_dist < 0.05

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        
        
        # Define cube size for reference (used for target zone sizing)
        self.cube_size = [0.021, 0.021, 0.021]  # Average of min/max
        
        self.cube = BoxObject(
            name="cube",
            size_min=[0.020, 0.020, 0.020],  # [0.015, 0.015, 0.015],
            size_max=[0.022, 0.022, 0.022],  # [0.018, 0.018, 0.018])
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )
        
        # Create target zone (twice the width and length of the cube, very thin)
        # This is a visual marker showing where to place the cube
        target_size = [self.cube_size[0] * 2, self.cube_size[1] * 2, 0.002]  # Thin flat square

        self.target_zone = BoxObject(
            name="target_zone",
            size_min=target_size,
            size_max=target_size,
            rgba=[1, 0, 0, 1],  
            material=redwood,
            joints=None,  # No joints makes it static/immovable
        )

        # Calculate table half-dimensions for clarity
        table_half_width = (self.table_full_size[0] / 2.0) - 0.2  # Adjusted to ensure cube is reachable
        table_half_depth = (self.table_full_size[1] / 2.0) - 0.2
        
        # Use SequentialCompositeSampler to place cube first, then target zone
        # This ensures the target zone doesn't spawn under the cube
        self.placement_initializer = SequentialCompositeSampler(
            name="ObjectSampler",
        )
        
        # Add cube sampler first
        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="CubeSampler",
                mujoco_objects=self.cube,
                x_range=[-table_half_width, table_half_width],
                y_range=[-table_half_depth, table_half_depth],
                rotation=None,
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
        )
        
        # Add target zone sampler second - placed on the table
        # SequentialCompositeSampler ensures target won't spawn under the cube
        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="TargetZoneSampler",
                mujoco_objects=self.target_zone,
                x_range=[-table_half_width, table_half_width],
                y_range=[-table_half_depth, table_half_depth],  # On the table surface
                rotation=None,
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,  # Ensures no overlap with previously placed objects (cube)
                reference_pos=self.table_offset,
                z_offset=0.001,  # Slightly above table surface
            )
        )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.cube, self.target_zone],
        )
        
    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)
        self.target_zone_body_id = self.sim.model.body_name2id(self.target_zone.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # define observables modality
            modality = "object"

            # cube-related observables
            @sensor(modality=modality)
            def cube_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube_body_id])

            @sensor(modality=modality)
            def cube_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")
            
            
            @sensor(modality=modality)
            def target_zone_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.target_zone_body_id])

            sensors = [cube_pos, cube_quat, target_zone_pos]

            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])

            # gripper to cube position sensor; one for each arm
            sensors += [
                self._get_obj_eef_sensor(full_pf, "cube_pos", f"{arm_pf}gripper_to_cube_pos", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )


        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self._was_grasping = False
        self._cube_was_lifted = False

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            cube_pos = None
            for obj_pos, obj_quat, obj in object_placements.values():
                if obj.joints is not None and len(obj.joints) > 0:
                    # Objects with joints: set joint position
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
                    # Store cube position for gripper positioning
                    if obj.name == "cube":
                        cube_pos = np.array(obj_pos)
                else:
                    # Static objects (no joints): set body position directly in the model
                    body_id = self.sim.model.body_name2id(obj.root_body)
                    self.sim.model.body_pos[body_id] = obj_pos
            
            # Always position gripper grasping the cube
            if cube_pos is not None:
                self._position_gripper_grasping_cube(cube_pos)

    def _position_gripper_grasping_cube(self, cube_pos):
        """
        Position the gripper so it is already grasping the cube at reset.
        Instead of complex IK, we move the gripper to a good grasping pose,
        then teleport the cube into the gripper.
        
        Args:
            cube_pos (np.array): 3D position of the cube [x, y, z]
        """
        import mujoco
        
        # Get the robot
        robot = self.robots[0]
        
        try:
            # Get the end effector site id
            arm = robot.arms[0] if hasattr(robot, 'arms') and len(robot.arms) > 0 else "right"
            gripper = robot.gripper[arm] if isinstance(robot.gripper, dict) else robot.gripper
            eef_site_name = gripper.important_sites["grip_site"]
            eef_site_id = self.sim.model.site_name2id(eef_site_name)
            
            # Get arm joint qpos indices from the robot
            arm_joint_qpos_indices = robot._ref_arm_joint_pos_indexes
            arm_joint_vel_indices = robot._ref_arm_joint_vel_indexes
            
            if len(arm_joint_qpos_indices) == 0:
                print("Warning: No arm joint indices found, skipping IK positioning")
                return
            
            # Target position: where we want the grip site to be
            # This should be at the cube center
            target_pos = cube_pos.copy()
            
            # Target orientation: gripper pointing straight down
            # The grip site Z-axis should point down (negative world Z)
            target_z_axis = np.array([0.0, 0.0, -1.0])  # Pointing down
            
            # IK parameters - tuned for reliable convergence
            max_iters = 500
            step_size = 0.2
            pos_tolerance = 0.001  # 1mm position tolerance
            ori_tolerance = 0.05   # Orientation tolerance (radians)
            damping = 0.05
            
            # Weight for position vs orientation
            pos_weight = 1.0
            ori_weight = 0.3
            
            for iteration in range(max_iters):
                # Forward to update state
                self.sim.forward()
                
                # Get current end effector position and orientation
                current_pos = self.sim.data.site_xpos[eef_site_id].copy()
                current_rot = self.sim.data.site_xmat[eef_site_id].reshape(3, 3)
                current_z_axis = current_rot[:, 2]  # Z-axis of gripper
                
                # Position error
                pos_error = target_pos - current_pos
                pos_error_norm = np.linalg.norm(pos_error)
                
                # Orientation error: we want gripper Z to align with target Z (pointing down)
                # Use cross product to get rotation axis, dot product for angle
                ori_error = np.cross(current_z_axis, target_z_axis)
                ori_error_norm = np.linalg.norm(ori_error)
                
                # Check convergence
                if pos_error_norm < pos_tolerance and ori_error_norm < ori_tolerance:
                    break
                
                # Get Jacobian for position and rotation
                jacp = np.zeros((3, self.sim.model.nv))
                jacr = np.zeros((3, self.sim.model.nv))
                mujoco.mj_jacSite(self.sim.model._model, self.sim.data._data, jacp, jacr, eef_site_id)
                
                # Extract Jacobian columns for arm joints only
                Jp = jacp[:, arm_joint_vel_indices]  # Position Jacobian
                Jr = jacr[:, arm_joint_vel_indices]  # Rotation Jacobian
                
                # Combine position and orientation into one task
                # Stack the Jacobians and errors
                J_combined = np.vstack([pos_weight * Jp, ori_weight * Jr])
                error_combined = np.concatenate([pos_weight * pos_error, ori_weight * ori_error])
                
                # Damped least squares
                JJT = J_combined @ J_combined.T
                damped_JJT = JJT + (damping ** 2) * np.eye(6)
                dq = J_combined.T @ np.linalg.solve(damped_JJT, step_size * error_combined)
                
                # Update arm joint positions
                for i, qpos_idx in enumerate(arm_joint_qpos_indices):
                    self.sim.data.qpos[qpos_idx] += dq[i]
                
                # Clip to joint limits
                for qpos_idx in arm_joint_qpos_indices:
                    # Get joint id from qpos address
                    for jnt_id in range(self.sim.model.njnt):
                        jnt_qpos_adr = self.sim.model.jnt_qposadr[jnt_id]
                        if jnt_qpos_adr == qpos_idx:
                            low, high = self.sim.model.jnt_range[jnt_id]
                            if low < high:  # Only clip if limits are defined
                                self.sim.data.qpos[qpos_idx] = np.clip(
                                    self.sim.data.qpos[qpos_idx], low, high
                                )
                            break
            
            # Close the gripper fingers first
            self._close_gripper_fingers(robot)
            
            # Forward pass to update state
            self.sim.forward()
            
            # NOW teleport the cube INTO the gripper
            # Get the current gripper position
            gripper_pos = self.sim.data.site_xpos[eef_site_id].copy()
            
            # Set cube position to be at the gripper grip site
            cube_joint_name = self.cube.joints[0] if self.cube.joints else None
            if cube_joint_name:
                # Get current cube quaternion (keep orientation)
                cube_quat = self.sim.data.body_xquat[self.cube_body_id].copy()
                # New cube position is at the gripper
                new_cube_pos = gripper_pos.copy()
                self.sim.data.set_joint_qpos(cube_joint_name, np.concatenate([new_cube_pos, cube_quat]))
            
            # Multiple forward passes to stabilize
            for _ in range(10):
                self.sim.forward()
            
            # Verify grasp
            final_pos = self.sim.data.site_xpos[eef_site_id].copy()
            final_cube_pos = self.sim.data.body_xpos[self.cube_body_id].copy()
            grasp_error = np.linalg.norm(final_pos - final_cube_pos)
            
            if grasp_error > 0.02:
                print(f"Warning: Cube not properly in gripper. Distance: {grasp_error:.4f}m")
            
        except Exception as e:
            # If IK fails, just print a warning and continue with default position
            print(f"Warning: Could not position gripper grasping cube via IK: {e}")
            import traceback
            traceback.print_exc()
    
    def _close_gripper_fingers(self, robot):
        """
        Close the gripper fingers to grasp an object.
        Sets the gripper joint positions to a closed/grasping state.
        
        Args:
            robot: The robot object with the gripper to close
        """
        try:
            # Get gripper joint indices - handle both dict and array formats
            gripper_qpos_indices_raw = robot._ref_gripper_joint_pos_indexes
            
            # Convert to flat list of indices
            if isinstance(gripper_qpos_indices_raw, dict):
                # Multi-arm robot: flatten all arm gripper indices
                gripper_qpos_indices = []
                for arm_indices in gripper_qpos_indices_raw.values():
                    if hasattr(arm_indices, '__iter__'):
                        gripper_qpos_indices.extend(list(arm_indices))
                    else:
                        gripper_qpos_indices.append(arm_indices)
            elif hasattr(gripper_qpos_indices_raw, '__iter__'):
                gripper_qpos_indices = list(gripper_qpos_indices_raw)
            else:
                gripper_qpos_indices = [gripper_qpos_indices_raw]
            
            if len(gripper_qpos_indices) == 0:
                return
            
            # For Panda gripper, the joints typically have range [0, 0.04]
            # where 0 = fully closed and 0.04 = fully open
            # Cube size is ~0.02m, so we need fingers slightly apart to grip it
            cube_half_width = self.cube_size[0] / 2  # ~0.01m
            grasp_position = cube_half_width + 0.002  # Slightly tighter than cube width
            
            for qpos_idx in gripper_qpos_indices:
                qpos_idx = int(qpos_idx)  # Ensure integer index
                # Get joint limits
                for jnt_id in range(self.sim.model.njnt):
                    jnt_qpos_adr = self.sim.model.jnt_qposadr[jnt_id]
                    if jnt_qpos_adr == qpos_idx:
                        low, high = self.sim.model.jnt_range[jnt_id]
                        if low < high:
                            self.sim.data.qpos[qpos_idx] = np.clip(grasp_position, low, high)
                        else:
                            self.sim.data.qpos[qpos_idx] = grasp_position
                        break
                else:
                    self.sim.data.qpos[qpos_idx] = grasp_position
            
            # Forward pass to update gripper state
            self.sim.forward()
            
        except Exception as e:
            print(f"Warning: Could not close gripper fingers: {e}")
            import traceback
            traceback.print_exc()

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube)

    def _check_success(self):
        """
        Check if cube has been successfully placed on the target zone.

        Returns:
            bool: True if cube is placed on target zone
        """
        return self._cube_on_target()
    
    def step(self, action):
        action = np.array(action, dtype=np.float32)
        
        obs, reward, done, info = super().step(action)
        
        if self._check_success():
            done = True
            info["success"] = True
        else:
            info["success"] = False
        return obs, reward, done, info