#!/usr/bin/env python3
import os
import site
import sys


def _strip_user_site_from_sys_path() -> None:
    os.environ['PYTHONNOUSERSITE'] = '1'
    user_site_markers = {
        os.path.realpath(site.getusersitepackages()),
        os.path.realpath(os.path.join(os.path.expanduser('~'), '.local')),
    }

    def _is_user_site_path(path: str) -> bool:
        if not path:
            return False
        real_path = os.path.realpath(path)
        return any(real_path.startswith(marker) for marker in user_site_markers if marker)

    sys.path[:] = [path for path in sys.path if not _is_user_site_path(path)]


_strip_user_site_from_sys_path()


def _purge_preloaded_mmcv_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == 'mmcv' or module_name.startswith('mmcv.'):
            sys.modules.pop(module_name, None)


_purge_preloaded_mmcv_modules()

HIPAD_ROOT = os.environ.get('HIPAD_ROOT', '/home/pnc/HiP-AD')
HIPAD_LEADERBOARD_ROOT = os.path.join(HIPAD_ROOT, 'bench2drive', 'leaderboard')
HIPAD_SCENARIO_RUNNER_ROOT = os.path.join(HIPAD_ROOT, 'bench2drive', 'scenario_runner')
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
PAD_LEADERBOARD_ROOT = os.path.join(REPO_ROOT, 'Bench2Drive', 'leaderboard')
DEFAULT_HIPAD_CHECKPOINT = os.path.join(HIPAD_ROOT, 'ckpts', 'hipad_stage2.pth')
DEFAULT_HIPAD_CONFIG = os.path.join(HIPAD_ROOT, 'projects', 'configs', 'hipad_b2d_stage2.py')
DEFAULT_HIPAD_SAVE_PATH = '/tmp/hipad_closed_loop_outputs'


def _append_sys_path(path: str) -> None:
    if path not in sys.path:
        sys.path.append(path)


def _setup_hipad_python_paths() -> None:
    carla_root = os.environ.get('CARLA_ROOT', '/home/pnc/carla_simulator')
    _append_sys_path(os.path.join(carla_root, 'PythonAPI'))
    _append_sys_path(os.path.join(carla_root, 'PythonAPI', 'carla'))
    _append_sys_path(REPO_ROOT)
    _append_sys_path(PAD_LEADERBOARD_ROOT)
    _append_sys_path(HIPAD_ROOT)
    _append_sys_path(HIPAD_LEADERBOARD_ROOT)
    _append_sys_path(HIPAD_SCENARIO_RUNNER_ROOT)
    os.environ.setdefault('CARLA_ROOT', carla_root)
    os.environ.setdefault('HIPAD_ROOT', HIPAD_ROOT)
    os.environ.setdefault('IS_BENCH2DRIVE', 'True')
    os.environ.setdefault('SAVE_PATH', DEFAULT_HIPAD_SAVE_PATH)


_setup_hipad_python_paths()

if REPO_ROOT in sys.path:
    sys.path.remove(REPO_ROOT)
from pad_team_code import hipad_adapter_agent as hipad_adapter_module
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

sys.modules['pad_team_code.pad_b2d_agent'] = hipad_adapter_module

import pad_closed_loop as base


ADAPTER_CHECKPOINT = hipad_adapter_module.DEFAULT_HIPAD_CHECKPOINT
ADAPTER_CONFIG = hipad_adapter_module.DEFAULT_HIPAD_CONFIG
hipadAdapterAgent = hipad_adapter_module.hipadAdapterAgent


base.padAgent = hipadAdapterAgent
base.DEFAULT_CHECKPOINT = ADAPTER_CHECKPOINT
base.DEFAULT_CONFIG = ADAPTER_CONFIG


class ClosedLoopRunner(base.ClosedLoopRunner):
    pass


def parse_args():
    args = base.parse_args()
    args.use_trajectory_post_processor = False
    return args


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f'checkpoint not found: {args.checkpoint}')
    if not os.path.exists(args.config):
        raise FileNotFoundError(f'config not found: {args.config}')
    if args.route_id is not None and not os.path.exists(args.routes):
        raise FileNotFoundError(f'routes xml not found: {args.routes}')

    runner = ClosedLoopRunner(args)
    try:
        runner.setup()
        runner.run()
    except KeyboardInterrupt:
        base.LOG.info('用户中断')
    finally:
        runner.cleanup()


if __name__ == '__main__':
    main()
