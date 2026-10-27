"""One-command operator launcher; help and dry-run never access the rig."""
import argparse
import json
import os
import subprocess
import sys

DEFAULT_PORT = '/dev/serial/by-id/usb-Teensyduino_USB_Serial_20242870-if00'


def _has(args, option):
    return any(x == option or x.startswith(option+'=') for x in args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    parser.add_argument('-h', '--help', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Print the launch plan; no serial, sudo or affinity changes.')
    parser.add_argument('--rt-priority', type=int, default=10, help='Child SCHED_RR priority 0..99; 0 explicitly disables realtime (default 10).')
    parser.add_argument('--no-affinity', action='store_true', help='Explicitly disable CPU partitioning.')
    parser.add_argument('--no-camera', action='store_true', help='Partition control CPUs without changing camera affinity.')
    parser.add_argument('--camera-service', default='cartpole-vision.service')
    parser.add_argument('--camera-cpus', default='0-3')
    parser.add_argument('--control-cpus', default='4-5')
    args, console_args = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if args.help:
        parser.print_help()
        print('\nConsole options (launcher defaults: port from CARTPOLE_PORT or installed Teensy, rate 250, keep board parameters):')
        from .console import main as console_main
        return console_main(['--help'])
    if not 0 <= args.rt_priority <= 99:
        parser.error('rt-priority must be 0..99')
    defaults = []
    if not _has(console_args, '--port'):
        defaults += ['--port', os.environ.get('CARTPOLE_PORT', DEFAULT_PORT)]
    if not _has(console_args, '--rate'):
        defaults += ['--rate', '250']
    if not _has(console_args, '--startup-profile') and not _has(console_args, '--keep-board-params'):
        defaults += ['--keep-board-params']
    console_args = defaults + console_args
    affinity_args = ([] if args.no_affinity else
        (['--no-camera'] if args.no_camera else ['--camera-service', args.camera_service, '--camera-cpus', args.camera_cpus])
        + ['--control-cpus', args.control_cpus])
    command = [sys.executable, '-m', 'cartpole_sdk.rig.policy_console', *console_args]
    if args.rt_priority:
        command = ['chrt', '--rr', str(args.rt_priority), *command]
    if args.dry_run:
        print(json.dumps(dict(console_args=console_args, affinity_args=affinity_args,
                              rt_priority=args.rt_priority, command=command)))
        return 0
    if not sys.platform.startswith('linux') and (affinity_args or args.rt_priority):
        parser.error('CPU/realtime launch requires Linux; use --no-affinity --rt-priority 0 for another host')
    try:
        if args.rt_priority:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_RTPRIO)
            if soft != resource.RLIM_INFINITY and soft < args.rt_priority:
                if hard == resource.RLIM_INFINITY or hard >= args.rt_priority:
                    resource.setrlimit(resource.RLIMIT_RTPRIO, (args.rt_priority, hard))
                else:
                    print('Requesting realtime allowance for this launcher only (console remains your user).', file=sys.stderr, flush=True)
                    subprocess.run(['sudo', 'prlimit', '--pid', str(os.getpid()),
                                    '--rtprio='+str(args.rt_priority)+':'+str(args.rt_priority)], check=True)
        if affinity_args:
            from .cpu_affinity import main as affinity_main
            return affinity_main([*affinity_args, '--', *command])
        os.execvp(command[0], command)
    except (OSError, subprocess.SubprocessError) as error:
        print('cartpole: '+str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
