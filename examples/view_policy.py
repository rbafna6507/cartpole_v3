"""Local actor/reference GUI; automatically uses mjpython on macOS."""
from cartpole_sdk.sim.policy_viewer import launch

if __name__ == '__main__':
    raise SystemExit(launch())
