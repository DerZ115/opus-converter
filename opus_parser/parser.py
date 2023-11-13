import logging
import re
from pathlib import Path
from typing import Union

from opus_parser.blocks import OpusDataBlock, OpusHeader, OpusParameterBlock

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# parameter_block_types = {
#     (16, 28): "Trace Parameters",
#     (31, 40): "Raman Parameters",
#     (32, 0):  "Instrument Parameters",
#     (48, 0):  "Acquisition Parameters",
#     (64, 0):  "FT Parameters",
#     (96, 0):  "Optics Parameters",
#     (160, 0): "Sample Parameters",
# }

metadata_block_types = {
    32:  "Instrument Parameters",
    48:  "Acquisition Parameters",
    64:  "FT Parameters",
    96:  "Optics Parameters",
    160: "Sample Parameters",
}

data_block_types = {
    "raman": (15, 40),
}

misc_blocks = {
    (0, 52, 0):  "Header",
    (0, 0, 160): "Unknown",
}

acquisition_modes = ["raman"]

measurement_block_types = {
    "raman": (31, 40),
}


class OpusFile:
    """Class for parsing single Opus files"""

    def __init__(self, path: Union[str, Path], mode: str = None) -> None:
        logger.debug('Initializing OpusFile instance')
        logger.debug(f'Received path: {path}')
        self.path = Path(path)
        self.acquisition_mode = mode
        self.header = None
        self.blocks = None

        self.bin_data = None
        self.data = None
        self.metadata = {}

    def _validate_file(self):
        """Make sure the file exists, and has a valid Opus file extension"""
        logger.debug('Validating parameters')
        if not self.path.exists():
            e = f'File {self.path} does not exist'
            logger.error(e)
            raise FileNotFoundError(e)
        if not self.path.is_file():
            e = f'Path {self.path} is not a file'
            logger.error(e)
            raise IsADirectoryError(e)
        if not re.match(r"\.\d+$", self.path.suffix):
            e = f"File {self.path} does not have a valid Opus file extension"
            logger.error(e)
            raise ValueError(e)

        logger.debug('Validation complete')

    def _get_metadata(self):
        logger.info('Parsing metadata blocks')
        metadata_blocks = [OpusParameterBlock(**block) for block in self.blocks
                           if block["block_type"][0] in metadata_block_types.keys()]

        for block in metadata_blocks:
            block_type = metadata_block_types[block.block_type[0]]
            block.read_data(self.bin_data)
            params = block.parse()
            if block_type in self.metadata.keys():
                if self.metadata[block_type] == params:
                    logger.info(f'Found duplicate metadata block: {block_type}, skipping...')
                    continue
                else:
                    logger.warning(f'Found conflicting metadata blocks: {block_type}, overwriting...')
                    # TODO: Ask user to resolve conflict
            self.metadata[block_type] = params

    def _check_acquisition_mode(self):
        """Check if the acquisition mode is valid"""
        if self.acquisition_mode is None:
            self._get_acquisition_mode()
        elif self.acquisition_mode not in acquisition_modes:
            e = f'Invalid acquisition mode: {self.acquisition_mode}'
            logger.error(e)
            raise ValueError(e)

    def _get_acquisition_mode(self):
        """Get the acquisition mode from the parameter blocks"""
        acq_mode = self.metadata['Acquisition Parameters']['AQM']
        if acq_mode == "RA":
            logger.info('Found Raman acquisition mode')
            self.acquisition_mode = "raman"

        else:
            e = f'Unknown acquisition mode: {acq_mode}'
            logger.error(e)
            raise ValueError(e)

    def _get_measurement_parameters(self):
        measurement_param_blocks = [OpusParameterBlock(**block).read_data(self.bin_data) for block in self.blocks
                                    if block["block_type"][:2] == measurement_block_types[self.acquisition_mode]]

        if len(measurement_param_blocks) > 1:
            if all(measurement_param_blocks[0].bin_data == block.bin_data for block in measurement_param_blocks):
                logger.info('Found duplicate measurement parameter blocks, skipping...')
            else:
                logger.warning(f'Found conflicting measurement parameters, using first one...')
                # TODO: Ask user to resolve conflict
        measurement_param_block = measurement_param_blocks[0]

        self.metadata["Measurement Parameters"] = measurement_param_block.parse()

    def _get_data(self):
        logger.info('Parsing data blocks')
        data_blocks = [OpusDataBlock(**block).read_data(self.bin_data) for block in self.blocks
                       if block["block_type"][:2] == data_block_types[self.acquisition_mode]]
        if len(data_blocks) > 1:
            if all(data_blocks[0].bin_data == block.bin_data for block in data_blocks):
                logger.info('Found duplicate data blocks, skipping...')
            else:
                logger.warning(f'Found conflicting data blocks, using first one...')
                # TODO: Ask user to resolve conflict
        data_block = data_blocks[0]

        data_parameters = self.metadata["Measurement Parameters"]
        num_points = data_parameters["NPT"]
        first_wn = data_parameters["FXV"]
        last_wn = data_parameters["LXV"]
        self.data = data_block.parse(num_points, first_wn, last_wn)

    def parse(self):
        """Parse the Opus file"""
        logger.info('Beginning parsing...')
        self._validate_file()

        logger.debug(f'Reading file {self.path}')
        with open(self.path, 'rb') as f:
            self.bin_data = f.read()

        logger.debug('Parsing header')
        self.header = OpusHeader(self.bin_data[:504])
        self.blocks = self.header.parse()

        self._get_metadata()
        self._check_acquisition_mode()

        self._get_measurement_parameters()
        self._get_data()

    def parse_metadata(self):
        self.metadata = {
            "path":             self.path,
            "acquisition_mode": self.acquisition_mode,
            "source":           self.metadata["Optics Parameters"].get("SRC"),
            "power":            self.metadata["Instrument Parameters"].get("RLP"),
            "aperture":         self.metadata["Optics Parameters"].get("APT"),
            "grating":          self.metadata["Instrument Parameters"].get(
                "GRN", self.metadata["Acquisition Parameters"].get("GRA")),
            "range":            self.metadata["Optics Parameters"].get("RNT"),
            "resolution":       self.metadata["Optics Parameters"].get("RST"),
            "objective":        self.metadata["Optics Parameters"].get("OBJ"),
            "integration time": self.metadata["Instrument Parameters"].get(
                "INT", self.metadata["Acquisition Parameters"].get("ITM", 0) / 1000),
            "accumulations":    self.metadata["Instrument Parameters"].get(
                "ASS", self.metadata["Acquisition Parameters"].get("NSS")),
            "date":             self.metadata["Raman Parameters"].get("DAT"),
            "time":             self.metadata["Raman Parameters"].get("TIM"),
            "sample name":      self.metadata["Sample Parameters"].get("SNM"),
            "sample form":      self.metadata["Sample Parameters"].get("SFM")
        }

        if self.metadata["integration time"] == 0:
            self.metadata["integration time"] = None

        return self.metadata


if __name__ == '__main__':
    from pprint import pprint

    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler('../opus_converter.log')
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(log_formatter)
    ch.setFormatter(log_formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    # opusfile = OpusFile("../test_files/test_raman_single.0")
    # opusfile = OpusFile("../test_files/test_raman_map.0")
    opusfile = OpusFile("../test_files/test_raman_map_2.0")
    # opusfile = OpusFile("../test_files/test_nir.0")
    opusfile.parse()
    # opusfile.parse_metadata()
    pprint(opusfile.metadata)
    print(opusfile.data)
