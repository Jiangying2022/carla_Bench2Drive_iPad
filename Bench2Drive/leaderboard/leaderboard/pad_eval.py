#!/usr/bin/env python
# Copyright (c) 2018-2019 Intel Corporation.
# authors: German Ros (german.ros@intel.com), Felipe Codevilla (felipe.alcm@gmail.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA Challenge Evaluator Routes

Provisional code to evaluate Autonomous Agents for the CARLA Autonomous Driving challenge
"""
import sys
import os
import glob
import collections
import collections.abc
import xml.etree.ElementTree as ET



if not hasattr(collections, 'Iterable'):
    collections.Iterable = collections.abc.Iterable

CARLA_ROOT=os.environ.get("CARLA_ROOT")+"/"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
BENCH2DRIVE_ZOO_ROOT = os.path.join(REPO_ROOT, 'Bench2DriveZoo')
NUPLAN_DEVKIT_ROOT = os.path.join(REPO_ROOT, 'nuplan-devkit')
bench2drive_root_default = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
Bench2Drive_ROOT = bench2drive_root_default
Bench2Drive_ROOT = Bench2Drive_ROOT + "/"

sys.path.append(REPO_ROOT)
sys.path.append(BENCH2DRIVE_ZOO_ROOT)
sys.path.append(NUPLAN_DEVKIT_ROOT)
sys.path.append(Bench2Drive_ROOT)

sys.path.append(CARLA_ROOT + "PythonAPI")
sys.path.append(CARLA_ROOT + "PythonAPI/carla")
carla_egg_paths = sorted(glob.glob(CARLA_ROOT + "PythonAPI/carla/dist/carla-0.9.16-*.egg"))
if not carla_egg_paths:
    carla_egg_paths = sorted(glob.glob(CARLA_ROOT + "PythonAPI/carla/dist/carla-*.egg"))
carla_whl_paths = sorted(glob.glob(CARLA_ROOT + "PythonAPI/carla/dist/carla-0.9.16-*.whl"))
if not carla_whl_paths:
    carla_whl_paths = sorted(glob.glob(CARLA_ROOT + "PythonAPI/carla/dist/carla-*.whl"))
carla_pkg_paths = carla_egg_paths if carla_egg_paths else carla_whl_paths
if carla_pkg_paths:
    sys.path.append(carla_pkg_paths[-1])

sys.path.append(Bench2Drive_ROOT + 'leaderboard')
sys.path.append(Bench2Drive_ROOT + 'leaderboard/pad_team_code')
sys.path.append(Bench2Drive_ROOT + 'scenario_runner')


def _get_available_map_towns():
    maps_dir = os.path.join(CARLA_ROOT, 'CarlaUE4', 'Content', 'Carla', 'Maps')
    map_paths = glob.glob(os.path.join(maps_dir, 'Town*.umap'))
    return {os.path.splitext(os.path.basename(path))[0] for path in map_paths}


def _filter_routes_file_for_available_maps(routes_file):
    available_towns = _get_available_map_towns()
    if not available_towns or not os.path.exists(routes_file):
        return routes_file

    tree = ET.parse(routes_file)
    root = tree.getroot()
    routes = list(root)
    supported_routes = [route for route in routes if route.attrib.get('town') in available_towns]

    if not supported_routes or len(supported_routes) == len(routes):
        return routes_file

    filtered_root = ET.Element(root.tag, root.attrib)
    for route in supported_routes:
        filtered_root.append(route)

    filtered_routes_file = os.path.join(Bench2Drive_ROOT, 'leaderboard', 'data', 'shuffle_available_maps.xml')
    ET.ElementTree(filtered_root).write(filtered_routes_file, encoding='utf-8', xml_declaration=True)
    return filtered_routes_file


default_routes = Bench2Drive_ROOT + "leaderboard/data/shuffle.xml"
requested_routes = os.environ.get("ROUTES_FILE") or os.environ.get("ROUTES") or default_routes
ROUTES = _filter_routes_file_for_available_maps(requested_routes) if requested_routes == default_routes else requested_routes

os.environ["Bench2Drive_ROOT"] = Bench2Drive_ROOT.rstrip('/')
os.environ["SAVE_PATH"] = Bench2Drive_ROOT+"/eval_pad/"

if not os.path.exists(os.environ.get("SAVE_PATH")):
    os.mkdir(os.environ.get("SAVE_PATH"))


visualize=False

if visualize:
    os.environ["TEAM_AGENT"]  = Bench2Drive_ROOT + "leaderboard/pad_team_code/pad_b2d_agent_vis.py"
else:
    os.environ["TEAM_AGENT"]= Bench2Drive_ROOT + "leaderboard/pad_team_code/pad_b2d_agent.py"

os.environ["IS_BENCH2DRIVE"] = "True"
os.environ["ROUTES"] = ROUTES
os.environ["CHECKPOINT_ENDPOINT"]=os.environ["SAVE_PATH"]+"eval.json"

os.environ["SCENARIO_RUNNER_ROOT"] = "scenario_runner"
os.environ["LEADERBOARD_ROOT"] = "leaderboard"

checkpoint_path = os.environ.get("CHECKPOINT_PATH") or ""

os.environ["TEAM_CONFIG"]=Bench2Drive_ROOT +"leaderboard/pad_team_code/pad_config.py+"+checkpoint_path

from leaderboard_evaluator import main

main()
