import argparse
import logging
import re
from pathlib import Path

from file import OpusFile

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def prep_logging(verbose: int) -> None:
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler('../opus_converter.log')
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    if verbose == 0:
        ch.setLevel(logging.WARNING)
    elif verbose == 1:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG)

    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(log_formatter)
    ch.setFormatter(log_formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parse Opus files')

    input_parser = parser.add_mutually_exclusive_group(required=True)
    input_parser.add_argument('-f', '--file', type=str, nargs='+', help='Path to Opus file(s)')
    input_parser.add_argument('-d', '--directory', type=str, help='Path to directory containing Opus files')

    parser.add_argument('-o', '--output', type=str, required=True, help='Path to output directory')
    parser.add_argument('-m', '--mode', type=str, help='Acquisition mode of the Opus file(s)')
    parser.add_argument('-v', '--verbose', action='count', default=0)

    args = parser.parse_args()

    prep_logging(args.verbose)

    if args.file:
        files = [Path(file) for file in args.file]
    else:
        files = [file for file in Path(args.directory).iterdir() if re.match(r"\.\d+$", file.suffix)]

    for file in files:
        opusfile = OpusFile(file, args.mode)
        opusfile.parse()
        opusfile.data.to_csv(Path(args.output, file.stem + '.csv'))
