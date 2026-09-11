"""AdaRoboVLG dynamic dexhand client."""

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
    """Build a dynamic dexterous-hand request from the selected packet."""
    request = load_request(args)
    # Override the recorded hand only when the user supplies an explicit option.
    if args.hand is not None:
        request["hand_name"] = args.hand
    request["mode"] = args.mode
    # The server selects the initial pose; later requests reuse its confirmed session ID.
    if args.mode == "track":
        request["tracking_id"] = args.tracking_id
    return request


def main():
    """Run the client dexhand command-line entrypoint."""
    parser = create_parser(__doc__, 5558)
    parser.add_argument(
        "--hand",
        choices=("AllegroHand", "InspireHand", "DH3Hand"),
        help="Override saved hand; omitted: packet or server default (right hand)",
    )
    add_tracking_arguments(parser)
    run_client(parse_args(parser), build_request)


if __name__ == "__main__":
    main()
