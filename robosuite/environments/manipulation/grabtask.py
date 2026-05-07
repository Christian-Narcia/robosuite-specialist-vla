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


class GrabTask(ManipulationEnv):
    """
    This class corresponds to the lifting task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        lite_physics (bool): Whether to optimize for mujoco forward and step calls to reduce total simulation overhead.
            Set to False to preserve backward compatibility with datasets collected in robosuite <= 1.4.1.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    # Task mode constants
    TASK_FULL = "full"          # Full pick and place task (default)
    TASK_REACH = "reach"        # End when gripper reaches cube
    TASK_GRASP = "grasp"        # End when gripper grasps cube

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
        task_mode="grasp",  # "full", "reach", or "grasp" - default to grasp
        reach_threshold=0.05,  # Distance threshold for reach task (meters)
        spawn_gripper_near_cube=True,  # If True, spawn gripper 0.07m from cube
        gripper_spawn_distance=0.08,  # Distance from cube to spawn gripper
        use_target_zone=True,  # If True, include green target platform
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
        
        # Task mode configuration
        assert task_mode in [self.TASK_FULL, self.TASK_REACH, self.TASK_GRASP], \
            f"Invalid task_mode '{task_mode}'. Must be one of: 'full', 'reach', 'grasp'"
        self.task_mode = task_mode
        self.reach_threshold = reach_threshold
        
        # Gripper spawn configuration
        self.spawn_gripper_near_cube = spawn_gripper_near_cube
        self.gripper_spawn_distance = gripper_spawn_distance
        
        # Target zone configuration
        self.use_target_zone = use_target_zone
        
        # Track previous distance for movement reward
        self._prev_cube_to_target_xy = None
        self._prev_cube_to_target_3d = None

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
        Reward function for grab task with strong incentives for grasping:
        - Reward for reaching the cube (distance-based)
        - Reward for gripper contact with cube
        - Reward for gripper closing when near cube
        - Large reward for successful grasp
        """
        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward = 2.25

        # use a shaping reward
        elif self.reward_shaping:

            # reaching reward
            dist = self._gripper_to_target(
                gripper=self.robots[0].gripper, target=self.cube.root_body, target_type="body", return_distance=True
            )
            reaching_reward = 1 - np.tanh(10.0 * dist)
            reward += reaching_reward

            # grasping reward
            if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube):
                reward += 0.25


        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        return reward

    
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
        self.target_zone = None
        if self.use_target_zone:
            target_size = [self.cube_size[0] * 2, self.cube_size[1] * 2, 0.002]  # Thin flat square
            redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
            self.target_zone = BoxObject(
                name="target_zone",
                size_min=target_size,
                size_max=target_size,
                rgba=[1, 0, 0, 1],  # Red color
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
        if self.use_target_zone:
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
        mujoco_objects = [self.cube]
        if self.use_target_zone:
            mujoco_objects.append(self.target_zone)
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=mujoco_objects,
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
        self.target_zone_body_id = None
        if self.use_target_zone:
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
            
            
            sensors = [cube_pos, cube_quat]
            
            if self.use_target_zone:
                @sensor(modality=modality)
                def target_zone_pos(obs_cache):
                    return np.array(self.sim.data.body_xpos[self.target_zone_body_id])
                sensors.append(target_zone_pos)

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
        # Reset previous distance tracking
        self._prev_cube_to_target_xy = None
        self._prev_cube_to_target_3d = None

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
            
            # Position gripper near the cube if enabled
            if self.spawn_gripper_near_cube and cube_pos is not None:
                self._position_gripper_near_cube(cube_pos)

    def _position_gripper_near_cube(self, cube_pos):
        """
        Position the gripper at a specified distance above the cube.
        Uses inverse kinematics to find joint positions that place the gripper near the cube.
        
        Args:
            cube_pos (np.array): 3D position of the cube [x, y, z]
        """
        import mujoco
        
        # Target position: directly above the cube at spawn distance
        target_pos = cube_pos.copy()
        target_pos[2] += self.gripper_spawn_distance  # Position above the cube
        
        # Get the robot
        robot = self.robots[0]
        
        try:
            # Get the end effector site id
            # For single arm robots, gripper is accessed via robot.gripper["right"] or similar
            arm = robot.arms[0] if hasattr(robot, 'arms') and len(robot.arms) > 0 else "right"
            gripper = robot.gripper[arm] if isinstance(robot.gripper, dict) else robot.gripper
            eef_site_name = gripper.important_sites["grip_site"]
            eef_site_id = self.sim.model.site_name2id(eef_site_name)
            
            # Get arm joint qpos indices from the robot
            # These are the indices into sim.data.qpos for the arm joints
            arm_joint_qpos_indices = robot._ref_arm_joint_pos_indexes
            
            if len(arm_joint_qpos_indices) == 0:
                print("Warning: No arm joint indices found, skipping IK positioning")
                return
            
            # IK parameters
            max_iters = 150
            step_size = 0.5
            tolerance = 0.003
            damping = 0.05  # Damping for damped least squares
            
            for iteration in range(max_iters):
                # Forward to update state
                self.sim.forward()
                
                # Get current end effector position
                current_pos = self.sim.data.site_xpos[eef_site_id].copy()
                
                # Position error
                pos_error = target_pos - current_pos
                error_norm = np.linalg.norm(pos_error)
                
                if error_norm < tolerance:
                    break
                
                # Get Jacobian for the end effector site
                jacp = np.zeros((3, self.sim.model.nv))
                jacr = np.zeros((3, self.sim.model.nv))
                mujoco.mj_jacSite(self.sim.model._model, self.sim.data._data, jacp, jacr, eef_site_id)
                
                # Extract Jacobian columns for arm joints only
                # We need qvel indices (velocity space), which for simple joints are the same as joint indices
                arm_joint_vel_indices = robot._ref_arm_joint_vel_indexes
                J = jacp[:, arm_joint_vel_indices]
                
                # Damped least squares (more stable than pseudoinverse)
                # dq = J^T (J J^T + λ²I)^(-1) * error
                JJT = J @ J.T
                damped_JJT = JJT + (damping ** 2) * np.eye(3)
                dq = J.T @ np.linalg.solve(damped_JJT, step_size * pos_error)
                
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
            
            # Final forward pass to update all states
            self.sim.forward()
            
            # Report final error
            final_pos = self.sim.data.site_xpos[eef_site_id].copy()
            final_error = np.linalg.norm(target_pos - final_pos)
            if final_error > 0.02:  # 2cm threshold for warning
                print(f"Warning: IK converged with error {final_error:.4f}m (target was {tolerance}m)")
            
        except Exception as e:
            # If IK fails, just print a warning and continue with default position
            print(f"Warning: Could not position gripper near cube via IK: {e}")
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

    def _check_reach(self):
        """
        Check if the gripper is close enough to the cube (reach task).
        
        Returns:
            bool: True if gripper is within reach_threshold of the cube
        """
        gripper_to_cube_dist = self._gripper_to_target(
            gripper=self.robots[0].gripper,
            target=self.cube.root_body,
            target_type="body",
            return_distance=True,
        )
        return gripper_to_cube_dist < self.reach_threshold
    
    def _check_cube_grasp(self):
        """
        Check if the gripper has successfully grasped the cube.
        
        Returns:
            bool: True if gripper is grasping the cube
        """
        return self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube)
    
    def _get_gripper_qpos(self):
        """
        Get the current gripper opening amount.
        For Panda gripper: 0 = fully closed, 0.04 = fully open
        
        Returns:
            float: Average gripper finger position, or None if not available
        """
        try:
            robot = self.robots[0]
            # Get gripper joint indices
            gripper_qpos_indices = robot._ref_gripper_joint_pos_indexes
            if len(gripper_qpos_indices) > 0:
                gripper_qpos = self.sim.data.qpos[gripper_qpos_indices]
                return np.mean(gripper_qpos)  # Average of both fingers
        except Exception:
            pass
        return None
    
    def _check_place(self):
        """
        Check if cube has been successfully placed on the target zone (full task).

        Returns:
            bool: True if cube is placed on target zone
        """
        return self._cube_on_target()

    def _check_success(self):
        """
        Check if the current task mode's success condition is met.
        
        Returns:
            bool: True if gripper has successfully grasped the cube
        """
        return self._check_cube_grasp()
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        if self._check_success():
            done = True

        return obs, reward, done, info