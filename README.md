## Example: Running Keyboard Robot Control

To collect data using the keyboard control script, run:

```sh
python robosuite/robosuite/demos/keyboard_robot_controlv2.py --environment MoveTask --robots Panda --cameras lbview rbview --show-pov-preview --preview-cameras lbview rbview --preview-scale 1.5
```

You can adjust the arguments for different environments, robots, cameras, and options as needed.

## Keyboard Controls

| Key            | Action                                      |
|----------------|---------------------------------------------|
| Z              | Discard current episode & restart           |
| Ctrl+Q         | End episode (saves if successful)           |
| Spacebar       | Open gripper                                |
| N              | Close gripper                               |
| Arrow Keys     | Move horizontally in x-y plane              |
| . (period)     | Move up (positive z)                        |
| ; (semicolon)  | Move down (negative z)                      |
| O / P          | Rotate yaw                                  |
| Y / H          | Rotate pitch                                |
| E / R          | Rotate roll                                 |
| S              | Switch active arm (multi-armed robots)      |
| =              | Switch active robot (multi-robot envs)      |

Refer to the script for more usage examples and options.
# robosuite-data-collection

## Installation & Environment Setup

### 1. Create Conda Environment


Create a new Conda environment with Python 3.12 (add `-y` to auto-confirm prompts):

```sh
conda create -n robosuite-collection python=3.12 -y
```

Activate the environment:

```sh
conda activate robosuite-collection
```


### 2. Install Python Dependencies

Change directory to robosuite and install requirements:

```sh
cd robosuite
pip install -r requirements.txt
```

### 3. Additional Requirements

You need additional requirements for data collection, install them with:

```sh
pip install -r ../data-collection-requirements.txt
```

This will install packages such as `h5py` and `torch` which are needed for data collection scripts.

---

