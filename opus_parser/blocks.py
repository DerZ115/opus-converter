import struct
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

field_types = {
    (0, 4): '<I',
    (1, 4): '<f',
    (1, 8): '<d'
}


class OpusHeader:
    """Class representing the header of an Opus file"""

    def __init__(self, header_data: bytes) -> None:
        logger.info('Initializing Header instance')
        if len(header_data) != 504:
            e = f'Invalid file header length: {len(header_data)}'
            logger.error(e)
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
        logger.debug(f'Found {len(blocks)} blocks in header')
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
        logger.debug('Reading data block')

        wavenumbers = np.linspace(first_wn, last_wn, num_points)

        if self.block_type[2] == 0:
            logger.debug('Found single spectrum in file')

            data_bin = self.bin_data[:4 * num_points]
            data = np.asarray(struct.unpack('<' + 'f' * num_points, data_bin)).reshape(1, -1)

        elif self.block_type[2] == 80:
            logger.debug('Found multiple spectra in file')
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
            logger.error(e)
            raise ValueError(e)

        data = pd.DataFrame(data, columns=wavenumbers)
        return data
