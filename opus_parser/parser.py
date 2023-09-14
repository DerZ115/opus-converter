import logging
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd

opus_logger = logging.getLogger(__name__)
opus_logger.addHandler(logging.NullHandler())

parameter_block_types = {
    (16, 28): "Trace Parameters",
    (31, 40):  "Raman Parameters",
    (32, 0):   "Instrument Parameters",
    (48, 0):   "Acquisition Parameters",
    (64, 0):   "FT Parameters",
    (96, 0):   "Optics Parameters",
    (160, 0):  "Sample Parameters",
}

data_block_types = {
    (15, 40):  "Raman Data",
}

misc_blocks = {
    (0, 52, 0):  "Header",
    (0, 0, 160): "Unknown",
}

unused_blocks = [7, 11, 23, 27]

field_types = {
    (0, 4): '<I',
    (1, 4): '<f',
    (1, 8): '<d'
}


class OpusFile:
    """Class for parsing single Opus files"""

    def __init__(self, path: Union[str, Path]) -> None:
        opus_logger.debug('Initializing OpusFile instance')
        opus_logger.debug(f'Received path: {path}')
        self.path = Path(path)
        self.acquisition_mode = None
        self.header = None
        self.parameter_blocks = None
        self.data_blocks = None
        self.misc_blocks = None

        self.bin_data = None
        self.data = None
        self.parameters = {}
        self.metadata = None

    def parse(self):
        """Parse the Opus file"""
        opus_logger.info('Beginning parsing...')
        self._validate_file()

        opus_logger.debug(f'Reading file {self.path}')
        with open(self.path, 'rb') as f:
            self.bin_data = f.read()

        opus_logger.debug('Parsing header')
        self.header = OpusHeader(self.bin_data[:504])
        block_list = self.header.parse()

        opus_logger.debug('Creating blocks')
        blocks = [self._make_block(**block) for block in block_list]
        self.parameter_blocks = [block for block in blocks if block.block_type[:2] in parameter_block_types.keys()]
        self.data_blocks = [block for block in blocks if block.block_type[:2] in data_block_types.keys()]
        self.misc_blocks = [block for block in blocks if block.block_type in misc_blocks.keys()]

        self._get_acquisition_mode()

        self._get_parameters()

        self._get_data()

    def _get_data(self):
        opus_logger.info('Parsing data blocks')
        for block in self.data_blocks:
            if self.acquisition_mode == "raman-single" and block.block_type == (15, 40, 0):
                data_parameters = self.parameters["Raman Parameters"]
                num_points = data_parameters["NPT"]
                first_wn = data_parameters["FXV"]
                last_wn = data_parameters["LXV"]
                self.data = block.parse(num_points, first_wn, last_wn)

            if self.acquisition_mode == "raman-multi" and block.block_type == (15, 40, 80):
                data_parameters = self.parameters["Raman Parameters"]
                num_points = data_parameters["NPT"]
                first_wn = data_parameters["FXV"]
                last_wn = data_parameters["LXV"]
                self.data = block.parse(num_points, first_wn, last_wn)

    def _get_parameters(self):
        opus_logger.info('Parsing parameter blocks')
        for block in self.parameter_blocks:
            param_name = parameter_block_types[block.block_type[:2]]
            params = block.parse()

            if param_name in self.parameters.keys():
                if self.parameters[param_name] == params:
                    opus_logger.info(f'Found duplicate parameter block: {param_name}, skipping...')
                    continue
                else:
                    opus_logger.warning(f'Found conflicting parameter blocks: {param_name}, overwriting...')
            self.parameters[param_name] = params

    def _validate_file(self):
        """Make sure the file exists, and has a valid Opus file extension"""
        opus_logger.debug('Validating parameters')
        if not self.path.exists():
            e = f'File {self.path} does not exist'
            opus_logger.error(e)
            raise FileNotFoundError(e)
        if not self.path.is_file():
            e = f'Path {self.path} is not a file'
            opus_logger.error(e)
            raise IsADirectoryError(e)
        if not re.match(r"\.\d+$", self.path.suffix):
            e = f"File {self.path} does not have a valid Opus file extension"
            opus_logger.error(e)
            raise ValueError(e)

        opus_logger.debug('Validation complete')
        return True

    def _make_block(self,
                    offset: int,
                    length: int,
                    block_type: tuple[int, int, int]) -> Union['OpusBlock', None]:

        if block_type[:2] in data_block_types.keys():
            opus_logger.debug(f'Found data block: {block_type}')
            return OpusDataBlock(offset,
                                 length,
                                 block_type).read_data(self.bin_data)
        elif block_type[:2] in parameter_block_types.keys():
            opus_logger.debug(f'Found parameter block: {block_type}')
            return OpusParameterBlock(offset,
                                      length,
                                      block_type).read_data(self.bin_data)
        else:
            opus_logger.debug(f'Found misc block: {block_type}')
            block = OpusBlock(offset,
                              length,
                              block_type).read_data(self.bin_data)
            if block_type not in misc_blocks.keys() and block_type[0] not in unused_blocks:
                opus_logger.debug(f'Unknown block type: {block_type}')
                opus_logger.debug(block)
            return block

    def _get_acquisition_mode(self):
        for block in self.parameter_blocks:
            if block.block_type == (48, 0, 0):
                block.parse()
                acq_mode = block.parameters['AQM']

                if acq_mode == "RA":
                    opus_logger.info('Found Raman acquisition mode')
                    if any([block.block_type == (31, 40, 0) for block in self.parameter_blocks]):
                        self.acquisition_mode = "raman-single"
                        opus_logger.info('Found single spectrum file')
                    elif any([block.block_type == (31, 40, 80) for block in self.parameter_blocks]):
                        self.acquisition_mode = "raman-multi"
                        opus_logger.info('Found multiple spectra file')

                else:
                    e = f'Unknown acquisition mode: {acq_mode}'
                    opus_logger.error(e)
                    raise ValueError(e)

    def parse_metadata(self):
        self.metadata = {
            "path":             self.path,
            "acquisition_mode": self.acquisition_mode,
            "source":           self.parameters["Optics Parameters"]["SRC"],
            "power":            self.parameters["Instrument Parameters"]["RLP"],
            "aperture":         self.parameters["Optics Parameters"]["APT"],
            "grating":          self.parameters["Optics Parameters"]["GRN"] or
                                self.parameters["Optics Parameters"]["GRA"],
            "integration time": self.parameters["Acquisition Parameters"]["INT"] or
                                self.parameters["Acquisition Parameters"]["ITM"] / 1000,
            "accumulations":    self.parameters["Acquisition Parameters"]["ASS"] or
                                self.parameters["Acquisition Parameters"]["NSS"],
            "date":             self.parameters["Raman Parameters"]["DAT"],
            "time":             self.parameters["Raman Parameters"]["TIM"],
        }


class OpusHeader:
    """Class representing the header of an Opus file"""

    def __init__(self, header_data: bytes) -> None:
        opus_logger.info('Initializing Header instance')
        if len(header_data) != 504:
            e = f'Invalid file header length: {len(header_data)}'
            opus_logger.error(e)
            raise ValueError(e)
        self.bin_data = header_data

    def parse(self) -> list[dict]:
        """Parse the header of the Opus file"""
        # Split into 12 byte blocks, ignore first 24 bytes
        blocks = []
        self.bin_data = [self.bin_data[i:i + 12] for i in range(24, len(self.bin_data), 12)]
        for block in self.bin_data:
            if block == b'\x00' * 12:  # Skip empty chunks
                continue
            blocks.append({
                "offset":     struct.unpack('<I', block[-4:])[0],
                "length":     4 * struct.unpack('<I', block[-8:-4])[0],
                "block_type": struct.unpack('<BBB', block[0:3])
            })
        opus_logger.debug(f'Found {len(blocks)} blocks in header')
        return blocks


@dataclass
class OpusBlock:
    """Class representing an Opus block, which can be either a parameter or data block"""
    offset: int
    length: int
    block_type: tuple[int, int, int]
    bin_data: bytes = b""

    def __str__(self):
        return self.__class__.__name__ + \
            f'(offset={self.offset}, length={self.length}, block_type={self.block_type})'

    def __repr__(self):
        return self.__str__()

    def read_data(self, bin_data: bytes):
        """Read the binary data from the Opus file"""
        self.bin_data = bin_data[self.offset:self.offset + self.length]
        return self


class OpusParameterBlock(OpusBlock):
    """Class representing an Opus parameter block"""

    def __init__(self,
                 offset: int,
                 length: int,
                 block_type: tuple[int, int, int]) -> None:
        super().__init__(offset, length, block_type)
        self.parameters = {}

    def parse(self) -> dict[str, Any]:
        """Parse the parameter block"""
        i = 0

        while i < len(self.bin_data):
            tag = self.bin_data[i:i + 3].decode('utf-8')
            if tag == 'END':
                break
            i += 4
            datatype = struct.unpack('<H', self.bin_data[i:i + 2])[0]
            byte_length = struct.unpack('<H', self.bin_data[i + 2:i + 4])[0] * 2
            i += 4
            if datatype >= 2:
                content = self.bin_data[i:i + byte_length].rstrip(b'\x00').decode('utf-8')
            else:
                content = struct.unpack(field_types[(datatype, byte_length)], self.bin_data[i:i + byte_length])[0]
            self.parameters[tag] = content
            i += byte_length

        return self.parameters


class OpusDataBlock(OpusBlock):
    """Class representing an Opus data block"""

    def __init__(self,
                 offset: int,
                 length: int,
                 block_type: tuple[int, int, int]) -> None:
        super().__init__(offset, length, block_type)

    def parse(self,
              num_points: int,
              first_wn: float,
              last_wn: float) -> pd.DataFrame:
        """Parse the data block"""
        opus_logger.debug('Reading data block')

        wavenumbers = np.linspace(first_wn, last_wn, num_points)

        if self.block_type[2] == 0:
            opus_logger.debug('Found single spectrum in file')

            data_bin = self.bin_data[:4 * num_points]
            data = np.asarray(struct.unpack('<' + 'f' * num_points, data_bin)).reshape(1, -1)

        elif self.block_type[2] == 80:
            opus_logger.debug('Found multiple spectra in file')
            header = struct.unpack('<' + 'I' * 4, self.bin_data[4:20])

            data = []
            ix = header[1]
            i = 0

            while i < header[0]:
                tmp_bin_data = self.bin_data[ix:ix + header[2]]
                data.append(np.asarray(struct.unpack('<' + 'f' * num_points, tmp_bin_data)))
                ix += header[2] + header[3]
                i += 1

            data = np.stack(data)

        else:
            e = f'Invalid data block type: {self.block_type}'
            opus_logger.error(e)
            raise ValueError(e)

        data = pd.DataFrame(data, columns=wavenumbers)
        return data


if __name__ == '__main__':
    from pprint import pprint
    opus_logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler('../opus_converter.log')
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(log_formatter)
    ch.setFormatter(log_formatter)

    opus_logger.addHandler(fh)
    opus_logger.addHandler(ch)

    opusfile = OpusFile("../test_files/test_raman.0")
    opusfile.parse()
    pprint(opusfile.parameters)
    print(opusfile.data)
