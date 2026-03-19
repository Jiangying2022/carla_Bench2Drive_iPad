#!/bin/bash
# Must set CARLA_ROOT
#export CARLA_ROOT=/mnt/workspace/navsim/exp/carla
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
CARLA_EGG=$(ls -1 ${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.16-*.egg 2>/dev/null | tail -n 1)
if [ -z "$CARLA_EGG" ]; then
    CARLA_EGG=$(ls -1 ${CARLA_ROOT}/PythonAPI/carla/dist/carla-*.egg 2>/dev/null | tail -n 1)
fi
CARLA_WHL=$(ls -1 ${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.16-*.whl 2>/dev/null | tail -n 1)
if [ -z "$CARLA_WHL" ]; then
    CARLA_WHL=$(ls -1 ${CARLA_ROOT}/PythonAPI/carla/dist/carla-*.whl 2>/dev/null | tail -n 1)
fi
if [ -n "$CARLA_EGG" ]; then
    export PYTHONPATH=$PYTHONPATH:$CARLA_EGG
elif [ -n "$CARLA_WHL" ]; then
    export PYTHONPATH=$PYTHONPATH:$CARLA_WHL
fi
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner
export SCENARIO_RUNNER_ROOT=scenario_runner
#
export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=$1
export TM_PORT=$2
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export RESUME=True
export IS_BENCH2DRIVE=$3
export PLANNER_TYPE=$9
export GPU_RANK=${10}
#
## TCP evaluation
export ROUTES=$4
export TEAM_AGENT=$5
export TEAM_CONFIG=$6
export CHECKPOINT_ENDPOINT=$7
export SAVE_PATH=$8

CUDA_VISIBLE_DEVICES=${GPU_RANK} python ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--traffic-manager-port=${TM_PORT} \
--gpu-rank=${GPU_RANK} \
