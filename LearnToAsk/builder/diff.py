import numpy as np, sys, random
from scipy.spatial import distance

# NOTE: THIS CODE NEEDS TO BE MAINTAINED FOR BOTH PYTHON 2 AND 3 COMPATIBILITY

build_region_specs = {
    "x_min_build": -5,
    "x_max_build": 5,
    "y_min_build": 1,
    "y_max_build": 9,
    "z_min_build": -5,
    "z_max_build": 5
}

all_possible_rot_values = [0, 90, 180, -90, -180]
rot_matrices_dict = {}
for rot_value in all_possible_rot_values:
    theta_yaw = np.radians(-1 * rot_value)
    c, s = np.cos(theta_yaw), np.sin(theta_yaw)
    R_yaw = np.matrix([ [ c, 0, -s ], [ 0, 1, 0 ], [ s, 0, c ] ])
    rot_matrices_dict[rot_value] = R_yaw

def is_feasible_next_placement(block, built_config, extra_check):
    # check if there is an existing block at block's location
    if extra_check and conflicting_block_exists(block, built_config):
        return False

    # check if block is on ground
    if block_on_ground(block):
        return True

    # check if block has a supporting block
    if block_with_support(block, built_config):
        return True
    else:
        return False

def conflicting_block_exists(block, built_config):
    for existing_block in built_config:
        if conflicts(existing_block, block):
            return True

    return False

def conflicts(existing_block, block):
    return existing_block["x"] == block["x"] and existing_block["y"] == block["y"] and existing_block["z"] == block["z"]

def block_on_ground(block):
    return block["y"] == 1

def block_with_support(block, built_config):
    for existing_block in built_config:
        if supports(existing_block, block):
            return True

    return False

def supports(existing_block, block):
    x_support = abs(existing_block["x"] - block["x"]) == 1 and existing_block["y"] == block["y"] and existing_block["z"] == block["z"]

    y_support = abs(existing_block["y"] - block["y"]) == 1 and existing_block["x"] == block["x"] and existing_block["z"] == block["z"]

    z_support = abs(existing_block["z"] - block["z"]) == 1 and existing_block["x"] == block["x"] and existing_block["y"] == block["y"]

    return x_support or y_support or z_support

def get_diff(gold_config, built_config, optimal_alignment=None):
    """
    Args:
        gold_config: Gold configuration
        built_config: Configuration built so far
        Both are lists of dicts. Each dict contains info on block type and block coordinates.

    Returns:
        A minimal diff in built config space -- in terms of placement and removal actions --
        to take the built config state to the goal config state

        All minimal diffs (each in both built and gold config space) with corresponding complementary info --
        complementary info would be the original built config, a perturbed config and the transformation to transform
        the former into the latter
    """

    # generate all possible perturbations of built config in the build region
    perturbations = generate_perturbations(built_config, gold_config = gold_config, optimal_alignment = optimal_alignment)

    # compute diffs for each perturbation
    diffs = list([diff(gold_config = gold_config, built_config = t.perturbed_config) for t in perturbations])

    # convert diffs back to actions in the built config space and not the perturbed config space
    # filter out perturbations that yield infeasible diff actions (those outside the build region)
    perturbations_and_diffs = list([x for x in list(zip(perturbations, diffs)) if is_feasible_perturbation(x[0], x[1])])

    # recompute diffs in gold config space
    # orig_diffs = list([diff(gold_config = gold_config, built_config = x[0].perturbed_config) for x in perturbations_and_diffs])
    # perturbations_diffs_and_orig_diffs = [x + (y,) for x, y in zip(perturbations_and_diffs, orig_diffs)]
    perturbations_and_diffs = list([(x[0], Diff(diff_built_config_space = x[1], diff_gold_config_space = None)) for x in perturbations_and_diffs])

    # select perturbation with min diff
    min_perturbation_and_diff = min(perturbations_and_diffs, key = lambda t: len(t[1].diff_built_config_space["gold_minus_built"]) + len(t[1].diff_built_config_space["built_minus_gold"]))

    # get all minimal diffs
    diff_sizes = list([len(t[1].diff_built_config_space["gold_minus_built"]) + len(t[1].diff_built_config_space["built_minus_gold"]) for t in perturbations_and_diffs])
    min_diff_size = min(diff_sizes)

    perturbations_and_diffs_and_diff_sizes = list(zip(perturbations_and_diffs, diff_sizes))
    perturbations_and_minimal_diffs_and_diff_sizes = list([x for x in perturbations_and_diffs_and_diff_sizes if x[1] == min_diff_size])

    # reformat final output
    perturbations_and_minimal_diffs = list([PerturbedConfigAndDiff(perturbed_config=x[0][0], diff=x[0][1]) for x in perturbations_and_minimal_diffs_and_diff_sizes])

    return min_perturbation_and_diff[1].diff_built_config_space, perturbations_and_minimal_diffs

def is_feasible_perturbation(perturbed_config, diff):
    # NOTE: This function mutates `diff`. DO NOT CHANGE THIS BEHAVIOR!
    """
    Args:
        perturbed_config: PerturbedConfig
        diff: Dict
    """

    def find_orig_block(block, block_pairs):
        return next(x[1] for x in block_pairs if x[0] == block)

    for key, diff_config in list(diff.items()):
        if key == "built_minus_gold": # retrieve from original built config instead of applying inverse transform
            block_pairs = list(zip(perturbed_config.perturbed_config, perturbed_config.original_config))
            diff[key] = list([find_orig_block(x, block_pairs) for x in diff_config])
        else:
            diff[key] = invert_perturbation_transform(
                config = diff_config,
                perturbed_config = perturbed_config
            )

    return is_feasible_config(diff["gold_minus_built"])

def is_feasible_config(config):
    """
    Args:
        config: List of blocks
    """
    def is_feasible_block(d):
        if (build_region_specs["x_min_build"] <= d["x"] <= build_region_specs["x_max_build"]) and (build_region_specs["y_min_build"] <= d["y"] <= build_region_specs["y_max_build"]) and (build_region_specs["z_min_build"] <= d["z"] <= build_region_specs["z_max_build"]):
            return True
        else:
            return False

    return all(is_feasible_block(block) for block in config)

def diff(gold_config, built_config):
    gold_config_reformatted = list(map(dict_to_tuple, gold_config))
    built_config_reformatted = list(map(dict_to_tuple, built_config))

    gold_minus_built = set(gold_config_reformatted) - set(built_config_reformatted)
    built_minus_gold = set(built_config_reformatted) - set(gold_config_reformatted)

    gold_minus_built = list(map(dict, gold_minus_built))
    built_minus_gold = list(map(dict, built_minus_gold))

    return {
        "gold_minus_built": gold_minus_built,
        "built_minus_gold": built_minus_gold
    }

def dict_to_tuple(d):
    return tuple(sorted(d.items()))

def generate_perturbations(config, gold_config, optimal_alignment=None):
    """
    Args:
        config: A configuration

    Returns:
        All perturbations of the config in build region
    """

    if optimal_alignment != None:
        perturbation = generate_perturbation(
            config, x_target=optimal_alignment[0], z_target=optimal_alignment[1], rot_target=optimal_alignment[2], gold_config=gold_config
        )

        return [perturbation]

    all_x_values = [i for i in range(build_region_specs["x_min_build"] - 5, build_region_specs["x_max_build"] + 1 + 5)]
    all_z_values = [i for i in range(build_region_specs["z_min_build"] - 5, build_region_specs["z_max_build"] + 1 + 5)]
    all_rot_values = [0, 90, 180, -90]

    perturbations = []

    for x in all_x_values:
        for z in all_z_values:
            for rot in all_rot_values:
                perturbation = generate_perturbation(config, x_target = x, z_target = z, rot_target = rot, gold_config = gold_config)
                perturbations.append(perturbation)

    return perturbations

def perturb_builder_position(dummy_block, x_target, z_target, rot_target):
    # move origin to x, z and with rotation

    # compute diff
    x_source = 0
    z_source = 0
    x_diff = x_target - x_source
    z_diff = z_target - z_source

    # translate
    def f(d, x_diff, z_diff):
        return {
            'x': d["x"] - x_diff,
            'y': d["y"],
            'z': d["z"] - z_diff,
            'type': d["type"]
        }

    dummy_block_translated = f(dummy_block, x_diff = x_diff, z_diff = z_diff)

    # rotate

    # rotate about origin

    # obtain yaw rotation matrix
    R_yaw = rot_matrices_dict[-1 * rot_target]

    def h(d, rot_matrix):
        v = np.matrix([ [ d["x"] ], [ d["y"] ], [ d["z"] ] ])
        v_new = rot_matrix * v

        return {
            'x': v_new.item(0),
            'y': v_new.item(1),
            'z': v_new.item(2),
            'type': d["type"]
        }

    dummy_block_translated_rotated = h(dummy_block_translated, rot_matrix = R_yaw)

    return dummy_block_translated_rotated

def generate_perturbation(config, x_target, z_target, rot_target, gold_config):

    if not config:
        # compute diff
        x_source = 0
        z_source = 0

        x_diff = x_target - x_source
        z_diff = z_target - z_source

        return PerturbedConfig(
            perturbed_config = [],
            x_target = x_target,
            z_target = z_target,
            rot_target = rot_target,
            rot_axis_pivot = (0, 0, 0),
            translation = (x_diff, z_diff),
            original_config = config
        )

    # move origin to x, z and with rotation

    # compute diff
    x_source = 0
    z_source = 0
    x_diff = x_target - x_source
    z_diff = z_target - z_source

    # translate
    def f(d, x_diff, z_diff):
        return {
            'x': d["x"] - x_diff,
            'y': d["y"],
            'z': d["z"] - z_diff,
            'type': d["type"]
        }

    config_translated = list([f(t, x_diff = x_diff, z_diff = z_diff) for t in config])

    # rotate

    # rotate about origin

    # obtain yaw rotation matrix
    R_yaw = rot_matrices_dict[-1 * rot_target]

    def h(d, rot_matrix):
        v = np.matrix([ [ d["x"] ], [ d["y"] ], [ d["z"] ] ])
        v_new = rot_matrix * v

        return {
            'x': int(round(v_new.item(0))),
            'y': int(round(v_new.item(1))),
            'z': int(round(v_new.item(2))),
            'type': d["type"]
        }

    config_translated_rotated = list([h(t, rot_matrix = R_yaw) for t in config_translated])

    return PerturbedConfig(
        perturbed_config = config_translated_rotated,
        x_target = x_target,
        z_target = z_target,
        rot_target = rot_target,
        rot_axis_pivot = (0, 0, 0),
        translation = (x_diff, z_diff),
        original_config = config
    )

def invert_perturbation_transform(config, perturbed_config):

    # rotate
    rot_target = -1 * perturbed_config.rot_target

    # rotate about pivot

    # obtain yaw rotation matrix
    R_yaw = rot_matrices_dict[-1 * rot_target]

    def h(d, rot_matrix):
        v = np.matrix([ [ d["x"] ], [ d["y"] ], [ d["z"] ] ])
        v_new = rot_matrix * v

        return {
            'x': int(round(v_new.item(0))),
            'y': int(round(v_new.item(1))),
            'z': int(round(v_new.item(2))),
            'type': d["type"]
        }

    config_rotated = list([h(t, rot_matrix = R_yaw) for t in config])

    x_diff = -1 * perturbed_config.translation[0]
    z_diff = -1 * perturbed_config.translation[1]

    # translate
    def f(d, x_diff, z_diff):
        return {
            'x': d["x"] - x_diff,
            'y': d["y"],
            'z': d["z"] - z_diff,
            'type': d["type"]
        }

    config_rotated_translated = list([f(t, x_diff = x_diff, z_diff = z_diff) for t in config_rotated])

    return config_rotated_translated

class PerturbedConfig:
    def __init__(self, perturbed_config, x_target, z_target, rot_target, rot_axis_pivot, translation, original_config):
        self.perturbed_config = perturbed_config
        self.rot_target = rot_target
        self.rot_axis_pivot = rot_axis_pivot
        self.translation = translation
        self.original_config = original_config
        self.x_target = x_target
        self.z_target = z_target

class Diff:
    def __init__(self, diff_built_config_space, diff_gold_config_space):
        self.diff_built_config_space = diff_built_config_space
        self.diff_gold_config_space = diff_gold_config_space

class PerturbedConfigAndDiff:
    def __init__(self, perturbed_config, diff):
        self.perturbed_config = perturbed_config
        self.diff = diff
