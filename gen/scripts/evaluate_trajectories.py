import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))

#sys.path.append('/home/uyeong/github/alfred')
#sys.path.append('/home/uyeong/github/alfred/gen')

import time
import multiprocessing as mp
import json
import random
import shutil
import argparse
import numpy as np
import pandas as pd
from collections import OrderedDict
from datetime import datetime

import constants
from agents.deterministic_planner_agent import DeterministicPlannerAgent
from env.thor_env import ThorEnv
from game_states.task_game_state_full_knowledge import TaskGameStateFullKnowledge
from utils.video_util import VideoSaver
from utils.dataset_management_util import load_successes_from_disk, load_fails_from_disk

# params
RAW_IMAGES_FOLDER = 'raw_images/'
DATA_JSON_FILENAME = 'traj_data.json'

# video saver
video_saver = VideoSaver()

# structures to help with constraint enforcement.
goal_to_required_variables = {"pick_and_place_simple": {"pickup", "receptacle", "scene"},
                              "pick_two_obj_and_place": {"pickup", "receptacle", "scene"},
                              "look_at_obj_in_light": {"pickup", "receptacle", "scene"},
                              "pick_clean_then_place_in_recep": {"pickup", "receptacle", "scene"},
                              "pick_heat_then_place_in_recep": {"pickup", "receptacle", "scene"},
                              "pick_cool_then_place_in_recep": {"pickup", "receptacle", "scene"},
                              "pick_and_place_with_movable_recep": {"pickup", "movable", "receptacle", "scene"}}
goal_to_pickup_type = {'pick_heat_then_place_in_recep': 'Heatable',
                       'pick_cool_then_place_in_recep': 'Coolable',
                       'pick_clean_then_place_in_recep': 'Cleanable'}
goal_to_receptacle_type = {'look_at_obj_in_light': "Toggleable"}
goal_to_invalid_receptacle = {'pick_heat_then_place_in_recep': {'Microwave'},
                              'pick_cool_then_place_in_recep': {'Fridge'},
                              'pick_clean_then_place_in_recep': {'SinkBasin'},
                              'pick_two_obj_and_place': {'CoffeeMachine', 'ToiletPaperHanger', 'HandTowelHolder'}}

scene_id_to_objs = {}
obj_to_scene_ids = {}
scenes_for_goal = {g: [] for g in constants.GOALS}
scene_to_type = {}


def sample_task_params(succ_traj, full_traj, fail_traj,
                       goal_candidates, pickup_candidates, movable_candidates, receptacle_candidates, scene_candidates,
                       inject_noise=10):
    # Get the current conditional distributions of all variables (goal/pickup/receptacle/scene).
    goal_weight = [(1 / (1 + np.random.randint(0, inject_noise + 1) + succ_traj.loc[
        (succ_traj['pickup'].isin(pickup_candidates) if 'pickup' in goal_to_required_variables[c] else True) &
        (succ_traj['movable'].isin(movable_candidates) if 'movable' in goal_to_required_variables[c] else True) &
        (succ_traj['receptacle'].isin(receptacle_candidates) if 'receptacle' in goal_to_required_variables[c] else True)
        & (succ_traj['scene'].isin(scene_candidates) if 'scene' in goal_to_required_variables[c] else True)]
            ['goal'].tolist().count(c)))  # Conditional.
                   * (1 / (1 + succ_traj['goal'].tolist().count(c)))  # Prior.
                   for c in goal_candidates]
    goal_probs = [w / sum(goal_weight) for w in goal_weight]

    pickup_weight = [(1 / (1 + np.random.randint(0, inject_noise + 1) +
                           sum([succ_traj.loc[
                                    succ_traj['goal'].isin([g]) &
                                    (succ_traj['movable'].isin(movable_candidates)
                                     if 'movable' in goal_to_required_variables[g] else True) &
                                    (succ_traj['receptacle'].isin(receptacle_candidates)
                                     if 'receptacle' in goal_to_required_variables[g] else True) &
                                    (succ_traj['scene'].isin(scene_candidates)
                                     if 'scene' in goal_to_required_variables[g] else True)]
                                ['pickup'].tolist().count(c) for g in goal_candidates])))
                     * (1 / (1 + succ_traj['pickup'].tolist().count(c)))
                     for c in pickup_candidates]
    pickup_probs = [w / sum(pickup_weight) for w in pickup_weight]

    movable_weight = [(1 / (1 + np.random.randint(0, inject_noise + 1) +
                            sum([succ_traj.loc[
                                     succ_traj['goal'].isin([g]) &
                                     (succ_traj['pickup'].isin(pickup_candidates)
                                      if 'pickup' in goal_to_required_variables[g] else True) &
                                     (succ_traj['receptacle'].isin(receptacle_candidates)
                                      if 'receptacle' in goal_to_required_variables[g] else True) &
                                     (succ_traj['scene'].isin(scene_candidates)
                                      if 'scene' in goal_to_required_variables[g] else True)]
                                 ['movable'].tolist().count(c) for g in goal_candidates])))
                      * (1 / (1 + succ_traj['movable'].tolist().count(c)))
                      for c in movable_candidates]
    movable_probs = [w / sum(movable_weight) for w in movable_weight]

    receptacle_weight = [(1 / (1 + np.random.randint(0, inject_noise + 1) +
                               sum([succ_traj.loc[
                                        succ_traj['goal'].isin([g]) &
                                        (succ_traj['pickup'].isin(pickup_candidates)
                                         if 'pickup' in goal_to_required_variables[g] else True) &
                                        (succ_traj['movable'].isin(movable_candidates)
                                         if 'movable' in goal_to_required_variables[g] else True) &
                                        (succ_traj['scene'].isin(scene_candidates)
                                         if 'scene' in goal_to_required_variables[g] else True)]
                                    ['receptacle'].tolist().count(c) for g in goal_candidates])))
                         * (1 / (1 + succ_traj['receptacle'].tolist().count(c)))
                         for c in receptacle_candidates]
    receptacle_probs = [w / sum(receptacle_weight) for w in receptacle_weight]
    scene_weight = [(1 / (1 + np.random.randint(0, inject_noise + 1) +
                          sum([succ_traj.loc[
                                   succ_traj['goal'].isin([g]) &
                                   (succ_traj['pickup'].isin(pickup_candidates)
                                    if 'pickup' in goal_to_required_variables[g] else True) &
                                   (succ_traj['movable'].isin(movable_candidates)
                                    if 'movable' in goal_to_required_variables[g] else True) &
                                   (succ_traj['receptacle'].isin(receptacle_candidates)
                                    if 'receptacle' in goal_to_required_variables[g] else True)]
                               ['scene'].tolist().count(c) for g in goal_candidates])))
                    * (1 / (1 + succ_traj['scene'].tolist().count(c)))
                    for c in scene_candidates]
    scene_probs = [w / sum(scene_weight) for w in scene_weight]

    # Calculate the probability difference between each value and the maximum so we can iterate over them to find a
    # next-best candidate to sample subject to the constraints of knowing which will fail.
    diffs = [("goal", goal_candidates[idx], goal_probs[idx] - min(goal_probs))
             for idx in range(len(goal_candidates)) if len(goal_candidates) > 1]
    diffs.extend([("pickup", pickup_candidates[idx], pickup_probs[idx] - min(pickup_probs))
                  for idx in range(len(pickup_candidates)) if len(pickup_candidates) > 1])
    diffs.extend([("movable", movable_candidates[idx], movable_probs[idx] - min(movable_probs))
                  for idx in range(len(movable_candidates)) if len(movable_candidates) > 1])
    diffs.extend([("receptacle", receptacle_candidates[idx], receptacle_probs[idx] - min(receptacle_probs))
                  for idx in range(len(receptacle_candidates)) if len(receptacle_candidates) > 1])
    diffs.extend([("scene", scene_candidates[idx], scene_probs[idx] - min(scene_probs))
                  for idx in range(len(scene_candidates)) if len(scene_candidates) > 1])

    # Iteratively pop the next biggest difference until we find a combination that is valid (e.g., not already
    # flagged as impossible by the simulator).
    variable_value_by_diff = {}
    diffs_as_keys = []  # list of diffs; index into list will be used as key values.
    for _, _, diff in diffs:
        already_keyed = False
        for existing_diff in diffs_as_keys:
            if np.isclose(existing_diff, diff):
                already_keyed = True
                break
        if not already_keyed:
            diffs_as_keys.append(diff)
    for variable, value, diff in diffs:
        key = None
        for kidx in range(len(diffs_as_keys)):
            if np.isclose(diffs_as_keys[kidx], diff):
                key = kidx
        if key not in variable_value_by_diff:
            variable_value_by_diff[key] = []
        variable_value_by_diff[key].append((variable, value))

    for key, diff in sorted(enumerate(diffs_as_keys), key=lambda x: x[1], reverse=True):
        variable_value = variable_value_by_diff[key]
        random.shuffle(variable_value)
        for variable, value in variable_value:

            # Select a goal.
            if variable == "goal":
                gtype = value
                # print("sampled goal '%s' with prob %.4f" % (gtype, goal_probs[goal_candidates.index(gtype)]))
                _goal_candidates = [gtype]

                _pickup_candidates = pickup_candidates[:]
                _movable_candidates = movable_candidates[:]
                _receptacle_candidates = receptacle_candidates[:]
                _scene_candidates = scene_candidates[:]

            # Select a pickup object.
            elif variable == "pickup":
                pickup_obj = value
                # print("sampled pickup object '%s' with prob %.4f" %
                #       (pickup_obj,  pickup_probs[pickup_candidates.index(pickup_obj)]))
                _pickup_candidates = [pickup_obj]

                _goal_candidates = goal_candidates[:]
                _movable_candidates = movable_candidates[:]
                _receptacle_candidates = receptacle_candidates[:]
                _scene_candidates = scene_candidates[:]

            # Select a movable object.
            elif variable == "movable":
                movable_obj = value
                # print("sampled movable object '%s' with prob %.4f" %
                #       (movable_obj,  movable_probs[movable_candidates.index(movable_obj)]))
                _movable_candidates = [movable_obj]
                _goal_candidates = [g for g in goal_candidates if g == 'pick_and_place_with_movable_recep']

                _pickup_candidates = pickup_candidates[:]
                _receptacle_candidates = receptacle_candidates[:]
                _scene_candidates = scene_candidates[:]

            # Select a receptacle.
            elif variable == "receptacle":
                receptacle_obj = value
                # print("sampled receptacle object '%s' with prob %.4f" %
                #       (receptacle_obj, receptacle_probs[receptacle_candidates.index(receptacle_obj)]))
                _receptacle_candidates = [receptacle_obj]

                _goal_candidates = goal_candidates[:]
                _pickup_candidates = pickup_candidates[:]
                _movable_candidates = movable_candidates[:]
                _scene_candidates = scene_candidates[:]

            # Select a scene.
            else:
                sampled_scene = value
                # print("sampled scene %s with prob %.4f" %
                #       (sampled_scene, scene_probs[scene_candidates.index(sampled_scene)]))
                _scene_candidates = [sampled_scene]

                _goal_candidates = goal_candidates[:]
                _pickup_candidates = pickup_candidates[:]
                _movable_candidates = movable_candidates[:]
                _receptacle_candidates = receptacle_candidates[:]
            # Perform constraint propagation to determine whether this is a valid assignment.
            propagation_finished = False
            while not propagation_finished:
                assignment_lens = (len(_goal_candidates), len(_pickup_candidates), len(_movable_candidates),
                                   len(_receptacle_candidates), len(_scene_candidates))
                # Constraints on goal.
                _goal_candidates = [g for g in _goal_candidates if
                                    (g not in goal_to_pickup_type or
                                     len(set(_pickup_candidates).intersection(  # Pickup constraint.
                                        constants.VAL_ACTION_OBJECTS[goal_to_pickup_type[g]])) > 0)
                                    and (g not in goal_to_receptacle_type or
                                         np.any([r in constants.VAL_ACTION_OBJECTS[goal_to_receptacle_type[g]]
                                                for r in _receptacle_candidates]))  # Valid by goal receptacle const.
                                    and (g not in goal_to_invalid_receptacle or
                                         len(set(_receptacle_candidates).difference(
                                            goal_to_invalid_receptacle[g])) > 0)  # Invalid by goal receptacle const.
                                    and len(set(_scene_candidates).intersection(
                                        scenes_for_goal[g])) > 0  # Scene constraint
                                    ]

                # Define whether to consider constraints for each role based on current set of candidate goals.
                pickup_constrained = np.any(["pickup" in goal_to_required_variables[g] for g in _goal_candidates])
                movable_constrained = np.any(["movable" in goal_to_required_variables[g] for g in _goal_candidates])
                receptacle_constrained = np.any(["receptacle" in goal_to_required_variables[g]
                                                 for g in _goal_candidates])
                scene_constrained = np.any(["scene" in goal_to_required_variables[g] for g in _goal_candidates])

                # Constraints on pickup obj.
                _pickup_candidates = [p for p in _pickup_candidates if
                                      np.any([g not in goal_to_pickup_type or
                                              p in constants.VAL_ACTION_OBJECTS[goal_to_pickup_type[g]]
                                              for g in _goal_candidates])  # Goal constraint.
                                      and (not movable_constrained or
                                           np.any([p in constants.VAL_RECEPTACLE_OBJECTS[m]
                                                  for m in _movable_candidates]))  # Movable constraint.
                                      and (not receptacle_constrained or
                                           np.any([r in constants.VAL_ACTION_OBJECTS["Toggleable"] or
                                                  p in constants.VAL_RECEPTACLE_OBJECTS[r]
                                                  for r in _receptacle_candidates]))  # Receptacle constraint.
                                      and (not scene_constrained or
                                           np.any([s in obj_to_scene_ids[constants.OBJ_PARENTS[p]]
                                                   for s in _scene_candidates])) # Scene constraint
                                      ]
                # Constraints on movable obj.
                _movable_candidates = [m for m in _movable_candidates if
                                       'pick_and_place_with_movable_recep' in _goal_candidates  # Goal constraint
                                       and (not pickup_constrained or
                                            np.any([p in constants.VAL_RECEPTACLE_OBJECTS[m]
                                                   for p in _pickup_candidates]))  # Pickup constraint.
                                       and (not receptacle_constrained or
                                            np.any([r in constants.VAL_RECEPTACLE_OBJECTS and
                                                   m in constants.VAL_RECEPTACLE_OBJECTS[r]
                                                   for r in _receptacle_candidates]))  # Receptacle constraint.
                                       and (not scene_constrained or
                                            np.any([s in obj_to_scene_ids[constants.OBJ_PARENTS[m]]
                                                    for s in _scene_candidates]))  # Scene constraint
                                       ]
                # Constraints on receptacle obj.
                _receptacle_candidates = [r for r in _receptacle_candidates if
                                          np.any([(g not in goal_to_receptacle_type or
                                                   r in constants.VAL_ACTION_OBJECTS[goal_to_receptacle_type[g]]) and
                                                  (g not in goal_to_invalid_receptacle or
                                                  r not in goal_to_invalid_receptacle[g])
                                                  for g in _goal_candidates])  # Goal constraint.
                                          and (not receptacle_constrained or
                                               r in constants.VAL_ACTION_OBJECTS["Toggleable"] or
                                               np.any([p in constants.VAL_RECEPTACLE_OBJECTS[r]
                                                       for p in _pickup_candidates]))  # Pickup constraint.
                                          and (not movable_constrained or
                                               r in constants.VAL_ACTION_OBJECTS["Toggleable"] or
                                               np.any([m in constants.VAL_RECEPTACLE_OBJECTS[r]
                                                       for m in _movable_candidates]))  # Movable constraint.
                                          and (not scene_constrained or
                                               np.any([s in obj_to_scene_ids[constants.OBJ_PARENTS[r]]
                                                       for s in _scene_candidates]))  # Scene constraint
                                          ]
                # Constraints on scene.
                _scene_candidates = [s for s in _scene_candidates if
                                     np.any([s in scenes_for_goal[g]
                                             for g in _goal_candidates])  # Goal constraint.
                                     and (not pickup_constrained or
                                          np.any([obj_to_scene_ids[constants.OBJ_PARENTS[p]]
                                                  for p in _pickup_candidates]))  # Pickup constraint.
                                     and (not movable_constrained or
                                          np.any([obj_to_scene_ids[constants.OBJ_PARENTS[m]]
                                                  for m in _movable_candidates]))  # Movable constraint.
                                     and (not receptacle_constrained or
                                          np.any([obj_to_scene_ids[constants.OBJ_PARENTS[r]]
                                                  for r in _receptacle_candidates]))  # Receptacle constraint.
                                     ]
                if assignment_lens == (len(_goal_candidates), len(_pickup_candidates), len(_movable_candidates),
                                       len(_receptacle_candidates), len(_scene_candidates)):
                    propagation_finished = True

            candidate_lens = {"goal": len(_goal_candidates), "pickup": len(_pickup_candidates),
                              "movable": len(_movable_candidates), "receptacle": len(_receptacle_candidates),
                              "scene": len(_scene_candidates)}
            if candidate_lens["goal"] == 0:
                # print("Goal over-constrained; skipping")
                continue
            if np.all([0 in [candidate_lens[v] for v in goal_to_required_variables[g]] for g in _goal_candidates]):
                continue

            # Ensure some combination of the remaining constraints is not in failures and is not already populated
            # by the target number of repeats.
            failure_ensured = True
            full_ensured = True
            for g in _goal_candidates:
                pickup_iter = _pickup_candidates if "pickup" in goal_to_required_variables[g] else ["None"]
                for p in pickup_iter:
                    movable_iter = _movable_candidates if "movable" in goal_to_required_variables[g] else ["None"]
                    for m in movable_iter:
                        receptacle_iter = _receptacle_candidates if "receptacle" in goal_to_required_variables[g] \
                            else ["None"]
                        for r in receptacle_iter:
                            scene_iter = _scene_candidates if "scene" in goal_to_required_variables[g] else ["None"]
                            for s in scene_iter:
                                if (g, p, m, r, s) not in fail_traj:
                                    failure_ensured = False
                                if (g, p, m, r, s) not in full_traj:
                                    full_ensured = False
                                if not failure_ensured and not full_ensured:
                                    break
                            if not failure_ensured and not full_ensured:
                                break
                        if not failure_ensured and not full_ensured:
                            break
                    if not failure_ensured and not full_ensured:
                        break
                if not failure_ensured and not full_ensured:
                    break
            if failure_ensured:
                continue
            if full_ensured:
                continue

            if candidate_lens["goal"] > 1 or np.any([np.any([candidate_lens[v] > 1
                                                             for v in goal_to_required_variables[g]])
                                                     for g in _goal_candidates]):
                task_sampler = sample_task_params(succ_traj, full_traj, fail_traj,
                                                  _goal_candidates, _pickup_candidates, _movable_candidates,
                                                  _receptacle_candidates, _scene_candidates)
                sampled_task = next(task_sampler)
                if sampled_task is None:
                    continue
            else:
                g = _goal_candidates[0]
                p = _pickup_candidates[0] if "pickup" in goal_to_required_variables[g] else "None"
                m = _movable_candidates[0] if "movable" in goal_to_required_variables[g] else "None"
                r = _receptacle_candidates[0] if "receptacle" in goal_to_required_variables[g] else "None"
                s = _scene_candidates[0] if "scene" in goal_to_required_variables[g] else "None"
                sampled_task = (g, p, m, r, int(s))

            yield sampled_task

    yield None  # Discovered that there are no valid assignments remaining.

def ans2x(ans):
    return ans.lower().replace('.', '').replace(',', '').replace('\'', '').replace('-', '').replace('/', '')

import pickle
import pprint
# Return pddls corresponding given annotations given in a given traj_data
def replaced_pddl(pretrained_file, traj_data):
    with open(pretrained_file, 'rb') as f:
        data = pickle.load(f)
        # pp = pprint.PrettyPrinter()
        # anns in given traj_data
    annsList = [ans2x(r["task_desc"]) for r in traj_data["turk_annotations"]["anns"]]
    pddls = []
    for ans in annsList:
        for anns, pddl in data.items():
            if anns == ans:
                pddls.append(pddl)
    if len(pddls) == 0:
        print("!!!! NO PDDLS AVAILABLE !!!!")
        
    for i in range(len(pddls)):
        if pddls[i]["mrecep_target"] is None:
            pddls[i]["mrecep_target"] = ""
        if pddls[i]["parent_target"] is None:
            pddls[i]["parent_target"] = ""
        if "sliced" in pddls[i].keys():
            pddls[i]["object_sliced"] = (pddls[i]["sliced"]==1)
            del pddls[i]["sliced"]
        if "object_sliced" not in pddls[i].keys():
            print(pddls[i].keys())
        if "toggle_target" in pddls[i].keys() \
            and pddls[i]["toggle_target"] is None:
            pddls[i]["toggle_target"] = ""
    return pddls

def print_successes(succ_traj):
    print("###################################\n")
    print("Successes: ")
    print(succ_traj)
    print("\n##################################")
    
def get_pddlMatch(pddl, task):
    for file in os.listdir('scripts'):
        if os.path.isfile(os.path.join('scripts', file)) and pddl in file:
            filename = os.path.join('scripts', file)
    with open(filename, 'r') as f:
        print('pddl match loaded from {}'.format(filename))
        pddl_match = json.load(f)
    pTask = []
    for t in task:
        if '\n' in t:
            pTask.extend([_t.strip() for _t in t.split('\n') if _t != ''])
        else:
            pTask.append(t.strip().replace('\b', ''))
    return [pddl_match[t] for t in pTask]


def main(args):
    # create env and agent
    env = ThorEnv()

    game_state = TaskGameStateFullKnowledge(env)
    agent = DeterministicPlannerAgent(thread_id=0, game_state=game_state)
    
    taskType = ['pick_cool_then_place_in_recep', 
        'pick_and_place_with_movable_recep', 
        'pick_and_place_simple', 
        'pick_two_obj_and_place',
        'pick_heat_then_place_in_recep',
        'look_at_obj_in_light',
        'pick_clean_then_place_in_recep']
        
    root = '../data/json_2.1.0/'
    data_folder = '../data/json_2.1.0_Recep:{distance}-{pddl}'.format(distance=args.distance, pddl=args.pddl)
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    exp_folder = 'exp/{distance}'.format(distance=args.distance)
    if not os.path.exists(exp_folder):
        os.mkdir(exp_folder)
    from tqdm import tqdm
    try:
        for split in ['valid_seen', 'valid_unseen']:
            for ep in tqdm(os.listdir(os.path.join(root, split)), desc=split):
                # update fail detail for every episode
                # if 'fail-{split}-film.json'.format(split=split) in os.listdir(exp_folder):
                #     with open(os.path.join(exp_folder, 'fail-{split}-film.json'.format(split=split)), 'r') as f:
                #         saved = json.load(f)
                # else:
                #     saved = {}
                sp = {}
                for trial in os.listdir(os.path.join(root, split, ep)): 
                    # known fails
                    # if os.path.join(ep, trial) in saved:
                    #     continue
                    # generated are passed
                    if split in os.listdir(data_folder)\
                        and ep in os.listdir(os.path.join(data_folder, split))\
                        and trial in os.listdir(os.path.join(data_folder, split, ep)):
                        continue
                    
                    traj_root = os.path.join(root, split, ep, trial)
                    traj_data = json.load(open(traj_root+'/traj_data.json', 'r'))
                    task =[ann['task_desc'] for ann in traj_data['turk_annotations']['anns']]
                    
                    # Apply pddl differ by task argument prediction method
                    if args.pddl == 'filmReproduced':
                        pddls = replaced_pddl('scripts/exp/'+split+'_e100_2.p', traj_data)
                    elif args.pddl in ['roberta', 'gpt2', 'bert']:
                        pddls = get_pddlMatch(args.pddl, task)
                    else:
                        pddls = [traj_data["pddl_params"]]

                    for pddl in pddls:
                        traj_data["pddl_params"] = pddl    
                        setup_data_dict()

                        #######################################################################################################
                        scene = traj_data['scene']
                        objs = traj_data['scene']['object_poses']
                        seed = traj_data['scene']['random_seed']
                        agent.reset(seed=seed, scene=scene, objs=objs, traj_data=traj_data)
                        #######################################################################################################

                        #######################################################################################################
                        seed = traj_data['scene']['random_seed']
                        scene = traj_data['scene']
                        info = None
                        try:
                            agent.setup_problem(dict(seed=seed, info=info, traj_data=traj_data), scene=scene)
                        except Exception as e:
                            traj = os.path.join(ep, trial)
                            print('Fail:', os.path.join(ep, trial), e)
                            error = {'error': str(e)}
                            sp[traj] = error
                            continue
                        #######################################################################################################

                        #######################################################################################################
                        print("Performing reset via thor_env API")
                        sampled_scene = traj_data['scene']['floor_plan']
                        env.reset(sampled_scene)

                        print("Performing restore via thor_env API")
                        object_poses = traj_data['scene']['object_poses']
                        object_toggles = traj_data['scene']['object_toggles']
                        dirty_and_empty = traj_data['scene']['dirty_and_empty']
                        env.restore_scene(object_poses, object_toggles, dirty_and_empty)

                        env.step(traj_data['scene']['init_action'])
                        #######################################################################################################
                        try:
                            terminal = False
                            while not terminal and agent.current_frame_count <= constants.MAX_EPISODE_LENGTH:
                                action_dict = agent.get_action(None)
                                agent.step(action_dict)
                                reward, terminal = agent.get_reward()
                            dump_data_dict(os.path.join(data_folder, split, ep, trial))
                            print('Success:', os.path.join(split, ep, trial))
                        except Exception as e:
                            traj = os.path.join(ep, trial)
                            try:
                                action = game_state.last_printed_action.split("\t")[0]
                            except:
                                action = 'None'
                            print('Fail:', traj, e)
                            error = {'error': str(e), 'action': action}
                            sp[traj] = error
                saved = {}
                saved.update(sp)
                with open(os.path.join(exp_folder, 'fail-{pddl}-{split}-film.json'.format(pddl=args.pddl, split=split)), 'w') as f:
                    json.dump(saved, f, indent=4)
                    
    except OSError as e:
        if e.errno in [107, 2]:
            command = 'umount -l uykim'.split()
            os.chdir('/home/sangbeom')
            os.environ['ns'] = 'q1wjsfoq%)$'
            os.system('echo $ns | sudo -S umount -l uykim')
            os.system('echo yy12345 | sshfs uyeong@172.26.36.23:/home/uyeong/ uykim -o password_stdin')
            os.chdir('/home/sangbeom/uykim/github/alfred/gen/')
            os.system('python scripts/evaluate_trajectories.py')
        else:
            raise e
                # save_video()


def create_dirs(gtype, pickup_obj, movable_obj, receptacle_obj, scene_num):
    task_id = 'trial_T' + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_name = '%s-%s-%s-%s-%d' % (gtype, pickup_obj, movable_obj, receptacle_obj, scene_num) + '/' + task_id

    constants.save_path = os.path.join(constants.DATA_SAVE_PATH, save_name, RAW_IMAGES_FOLDER)
    if not os.path.exists(constants.save_path):
        os.makedirs(constants.save_path)

    print("Saving images to: " + constants.save_path)
    return task_id


def save_video():
    images_path = constants.save_path + '*.png'
    video_path = os.path.join(constants.save_path.replace(RAW_IMAGES_FOLDER, ''), 'video.mp4')
    video_saver.save(images_path, video_path)


def setup_data_dict():
    constants.data_dict = OrderedDict()
    constants.data_dict['task_id'] = ""
    constants.data_dict['task_type'] = ""
    constants.data_dict['scene'] = {'floor_plan': "", 'random_seed': -1, 'scene_num': -1, 'init_action': [],
                                    'object_poses': [], 'dirty_and_empty': None, 'object_toggles': []}
    constants.data_dict['plan'] = {'high_pddl': [], 'low_actions': []}
    constants.data_dict['images'] = []
    constants.data_dict['template'] = {'task_desc': "", 'high_descs': []}
    constants.data_dict['pddl_params'] = {'object_target': -1, 'object_sliced': -1,
                                          'parent_target': -1, 'toggle_target': -1,
                                          'mrecep_target': -1}
    constants.data_dict['dataset_params'] = {'video_frame_rate': -1}
    constants.data_dict['pddl_state'] = []


def dump_data_dict(path):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, DATA_JSON_FILENAME), 'w') as fp:
        json.dump(constants.data_dict, fp, sort_keys=True, indent=4)


def delete_save(in_parallel):
    save_folder = constants.save_path.replace(RAW_IMAGES_FOLDER, '')
    if os.path.exists(save_folder):
        try:
            shutil.rmtree(save_folder)
        except OSError as e:
            if in_parallel:  # another thread succeeded at this task while this one failed.
                return False
            else:
                raise e  # if we're not running in parallel, this is an actual.
    return True


def parallel_main(args):
    procs = [mp.Process(target=main, args=(args,)) for _ in range(args.num_threads)]
    try:
        for proc in procs:
            proc.start()
            time.sleep(0.1)
    finally:
        for proc in procs:
            proc.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument('--force_unsave', action='store_true', help="don't save any data (for debugging purposes)")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save_path', type=str, default="dataset/new_trajectories", help="where to save the generated data")
    parser.add_argument('--x_display', type=str, required=False, default=constants.X_DISPLAY, help="x_display id")
    parser.add_argument("--just_examine", action='store_true', help="just examine what data is gathered; don't gather more")
    parser.add_argument("--in_parallel", action='store_true', help="this collection will run in parallel with others, so load from disk on every new sample")
    parser.add_argument("-n", "--num_threads", type=int, default=1, help="number of processes for parallel mode")
    parser.add_argument('--json_file', type=str, default="", help="path to json file with trajectory dump")
    parser.add_argument('--distance', type=str, default="original", help="distance thres", choices=['0.3', 'avg', 'closest', 'original'])
    parser.add_argument('--pddl', type=str, default='original', help="pddl", choices=['original', 'filmReproduced', 'roberta', 'bert', 'gpt2'])
    # params
    parser.add_argument("--repeats_per_cond", type=int, default=3)
    parser.add_argument("--trials_before_fail", type=int, default=5)
    parser.add_argument("--async_load_every_n_samples", type=int, default=10)

    parse_args = parser.parse_args()

    if parse_args.in_parallel and parse_args.num_threads > 1:
        parallel_main(parse_args)
    else:
        main(parse_args)
