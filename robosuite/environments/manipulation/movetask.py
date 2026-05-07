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
        Simple reward based on cube-to-target distance with penalty for dropping.
        """
        reward = 0.0

        # --- State ---
        grasping_cube = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube)
        cube_pos = np.array(self.sim.data.body_xpos[self.cube_body_id])
        target_pos = np.array(self.sim.data.body_xpos[self.target_zone_body_id])
        
        # Distance from cube to target (3D)
        dist = np.linalg.norm(cube_pos - target_pos)
        
        # Distance-based reward: closer to target = higher reward
        # Use negative distance so closer is better, scaled for reasonable range
        reward = -dist
        
        # Penalty for dropping the cube
        if not grasping_cube:
            reward -= 2.0
        
        cube_pos = self.sim.data.body_xpos[self.cube_body_id]
        target_pos = self.sim.data.body_xpos[self.target_zone_body_id]

        # Simple distance check
        dist = np.linalg.norm(cube_pos[:2] - target_pos[:2])  # XY distance
        # print(f"distance from cube to target: {dist:.4f}m")
        # Bonus for success
        if self._check_success():
            reward += 10.0

        # --- Scale ---
        if self.reward_scale is not None:
            reward *= self.reward_scale

        return reward

    def _cube_to_target_distance(self):
        """
        Calculate the horizontal distance between the cube and the target zone center.

        Returns:
            float: Euclidean distance in the XY plane between cube and target zone
        """
        cube_pos = self.sim.data.body_xpos[self.cube_body_id]
        target_pos = self.sim.data.body_xpos[self.target_zone_body_id]
        # Only consider XY distance (horizontal plane)
        return np.linalg.norm(cube_pos[:2] - target_pos[:2])

    def _cube_on_target(self):
        """
        Check if the cube is at the target zone.

        Returns:
            bool: True if cube is within threshold distance of target
        """
        cube_pos = self.sim.data.body_xpos[self.cube_body_id]
        target_pos = self.sim.data.body_xpos[self.target_zone_body_id]

        # Simple distance check
        dist = np.linalg.norm(cube_pos[:2] - target_pos[:2])  # XY distance
        # print(f"distance from cube to target: {dist:.4f}m")
        
        return dist < 0.07  

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

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                if obj.joints is not None and len(obj.joints) > 0:
                    # Objects with joints: set joint position
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
                else:
                    # Static objects (no joints): set body position directly in the model
                    body_id = self.sim.model.body_name2id(obj.root_body)
                    self.sim.model.body_pos[body_id] = obj_pos

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
        obs, reward, done, info = super().step(action)
        if self._check_success():
            done = True
            info["success"] = True
        else:
            info["success"] = False
        return obs, reward, done, info