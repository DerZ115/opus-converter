import logging
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np

opus_logger = logging.getLogger(__name__)
opus_logger.addHandler(logging.NullHandler())

parameter_block_types = {
    48:  "Acquisition Parameters",
    64:  "FT Parameters",
    96:  "Optical Parameters",
    160: "Sample Parameters",
    31:  "{Channel} Parameters",
    32:  "Instrument Parameters"
}
data_block_types = {
    15: "{Channel} Data"
}

data2parameter_types = {
    15: 31
}

block_channels = {
    40: "Raman"
}

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
        self.header = None
        self.parameter_blocks = None
        self.data_blocks = None
        self.misc_blocks = None

        self.bin_data = None
        self.data = None
        self.metadata = None

    def parse(self):
        """Parse the Opus file"""
        opus_logger.debug('Beginning parsing...')
        self._validate_file()

        with open(self.path, 'rb') as f:
            self.bin_data = f.read()

        self.header = OpusHeader(self.bin_data[:504])
        block_list = self.header.parse()
        blocks = [self._make_block(**block) for block in block_list]
        self.parameter_blocks = [block for block in blocks if isinstance(block, OpusParameterBlock)]
        self.data_blocks = [block for block in blocks if isinstance(block, OpusDataBlock)]
        self.misc_blocks = [block for block in blocks if not isinstance(block, (OpusParameterBlock, OpusDataBlock))]

        num_points = {}
        for block in self.parameter_blocks:
            params = block.parse()
            if "NPT" in params.keys():
                num_points[(block.type, block.channel)] = params["NPT"]

        for block in self.data_blocks:
            block.parse(num_points[(data2parameter_types[block.type], block.channel)])

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
            raise NotADirectoryError(e)
        if not re.match(r"\.\d+$", self.path.suffix):
            e = f"File {self.path} does not have a valid Opus file extension"
            opus_logger.error(e)
            raise ValueError(e)

    def _make_block(self,
                    offset: int,
                    length: int,
                    block_type: int,
                    block_channel: int,
                    block_text: int) -> Union['OpusBlock', None]:
        if block_type in data_block_types.keys():
            return OpusDataBlock(offset,
                                 length,
                                 block_type,
                                 block_channel,
                                 block_text).read_data(self.bin_data)
        elif block_type in parameter_block_types.keys():
            return OpusParameterBlock(offset,
                                      length,
                                      block_type,
                                      block_channel,
                                      block_text).read_data(self.bin_data)
        else:
            block = OpusBlock(offset,
                              length,
                              block_type,
                              block_channel,
                              block_text).read_data(self.bin_data)
            opus_logger.warning(f'Unknown block type: {block_type}')
            opus_logger.debug(block)
            return block


class OpusHeader:
    """Class representing the header of an Opus file"""

    def __init__(self, header_data: bytes) -> None:
        opus_logger.info('Initializing Header instance')
        if len(header_data) != 504:
            e = f'Invalid file header length: {len(header_data)}'
            opus_logger.error(e)
            raise ValueError(e)
        self.bin_data = header_data

    def parse(self) -> list[dict[str, int]]:
        """Parse the header of the Opus file"""
        # Split into 12 byte blocks, ignore first 24 bytes
        blocks = []
        self.bin_data = [self.bin_data[i:i + 12] for i in range(24, len(self.bin_data), 12)]
        for block in self.bin_data:
            if block == b'\x00' * 12:  # Skip empty chunks
                continue
            blocks.append({"offset":        struct.unpack('<I', block[-4:])[0],
                           "length":        4 * struct.unpack('<I', block[-8:-4])[0],
                           "block_type":    struct.unpack('<B', block[0:1])[0],
                           "block_channel": struct.unpack('<B', block[1:2])[0],
                           "block_text":    struct.unpack('<B', block[2:3])[0]})
        return blocks


@dataclass
class OpusBlock:
    """Class representing an Opus block, which can be either a parameter or data block"""
    offset: int
    length: int
    type: int
    channel: int
    text: int
    bin_data: Union[bytes, None] = None

    def __str__(self):
        return self.__class__.__name__ + \
            f'(offset={self.offset}, length={self.length}, type={self.type}, channel={self.channel}, text={self.text})'

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
                 block_type: int,
                 block_channel: int,
                 block_text: int) -> None:
        super().__init__(offset, length, block_type, block_channel, block_text)
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
                 block_type: int,
                 block_channel: int,
                 block_text: int) -> None:
        super().__init__(offset, length, block_type, block_channel, block_text)
        self.data: Union[np.ndarray, None] = None

    def parse(self, num_points: int) -> np.ndarray:
        """Parse the data block"""
        opus_logger.debug('Reading data block')

        if len(self.bin_data) > (num_points + 1) * 4:
            opus_logger.debug('Found multiple spectra in file')
            # TODO: Implement multiple spectra files
            e = 'Multiple spectra files are not yet supported'
            opus_logger.error(e)
            raise NotImplementedError(e)
        else:
            opus_logger.debug('Found single spectrum in file')

            data_bin = self.bin_data[:4 * num_points]
            self.data = np.asarray(struct.unpack('<' + 'f' * num_points, data_bin)).reshape(1, -1)

        return self.data


if __name__ == '__main__':
    opus_logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler('../opus_converter.log')
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(log_formatter)
    ch.setFormatter(log_formatter)

    opus_logger.addHandler(fh)
    opus_logger.addHandler(ch)

    parser = OpusParser.from_dir('../data/', metadata=False, recursive=False)
    # parser = OpusParser("/home/daniel/opus-converter/data/230329_NP_fumarat.4")
    parser.parse()
    parser.export_data('../out/')
