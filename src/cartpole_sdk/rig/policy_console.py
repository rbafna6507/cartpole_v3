"""Run the operator console with a scoped 1 ms Python GIL switch interval."""
import sys


def main(argv=None, *, rig_factory=None, input_fn=None, output=None):
    """Delegate unchanged console arguments and restore the caller's interval."""
    previous_interval = sys.getswitchinterval()
    try:
        sys.setswitchinterval(.001)
        from . import console

        return console.main(argv, rig_factory=rig_factory, input_fn=input_fn, output=output)
    finally:
        sys.setswitchinterval(previous_interval)


if __name__ == '__main__':
    raise SystemExit(main())
