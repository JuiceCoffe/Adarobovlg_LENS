"""AdaRoboVLG dynamic gripper client."""

import sys
from pathlib import Path

# Allow direct script execution from the repository root.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client.common import (
    add_tracking_arguments,
    create_parser,
    load_request,
    parse_args,
    run_client,
)


def build_request(args):
    """Build a dynamic parallel-jaw request from the selected packet."""
    request = load_request(args)
    request.pop("hand_name", None)
    request["mode"] = args.mode
    # The server selects the initial pose; later requests reuse its confirmed session ID.
    if args.mode == "track":
        request["tracking_id"] = args.tracking_id
    return request


def main():
    """Run the client gripper command-line entrypoint."""
    parser = create_parser(__doc__, 5557)
    add_tracking_arguments(parser)
    run_client(parse_args(parser), build_request)


if __name__ == "__main__":
    main()
