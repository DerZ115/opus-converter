import re
import struct
from collections import Counter
from pathlib import Path
import logging

import numpy as np
import pandas as pd

opus_logger = logging.getLogger(__name__)
opus_logger.addHandler(logging.NullHandler())


class OpusParser(object):
    data: list | np.ndarray | pd.DataFrame
    params: list | pd.DataFrame
    metadata: bool | pd.DataFrame

    channel_dict = {
        "raman": 40
    }

    type_dict = {
        (0, 4): '<I',
        (1, 4): '<f',
        (1, 8): '<d'
    }

    def __init__(self, files, signal="raman", metadata=False, _basepath=Path()):
        opus_logger.info('Initializing OpusParser instance')
        self.files = files
        self.signal = signal
        self.store_metadata = metadata
        self._basepath = _basepath
        opus_logger.debug(f'Signal of type \'{signal}\' to be extracted')
        opus_logger.debug('Metadata will be output into separate file:', metadata)
        opus_logger.debug('Base path for output structure:', _basepath)
        # Prepare attributes
        self._bin_data = None
        self.params = []
        self.data = []

    def _validate_params(self):
        opus_logger.debug('Validating parameters')
        if isinstance(self.files, str):
            self.files = [self.files]
        self.files = [Path(file) for file in self.files]
        opus_logger.debug(f'Received {len(self.files)} files')

        for file in self.files:
            if not file.exists():
                raise FileNotFoundError(f"File {file} does not exist.")

        if self.signal not in self.channel_dict.keys():
            raise ValueError("Unknown signal type")

    # noinspection PyMethodMayBeStatic
    def _parse_header(self):
        opus_logger.debug('Reading file header')
        header = self._bin_data[24:504]
        header = [header[i:i + 12] for i in range(0, len(header), 12)]

        chunks = []
        for chunk in header:
            if chunk == b'\x00' * 12:
                break
            chunks.append({"offset": struct.unpack('<I', chunk[-4:])[0],
                           "length": struct.unpack('<I', chunk[-8:-4])[0],
                           "block": struct.unpack('<B', chunk[0:1])[0],
                           "channel": struct.unpack('<B', chunk[1:2])[0],
                           "type": struct.unpack('<B', chunk[2:3])[0]})

        opus_logger.debug(f'Found {len(chunks)} data chunks')
        return pd.DataFrame(chunks)

    def _create_masks(self, chunks):

        data_mask = chunks.block == 15
        param_mask = chunks.block == 31
        acquisition_mask = chunks.block == 32
        optics_mask = chunks.block == 96
        sample_mask = chunks.block == 160
        channel_mask = chunks.channel == self.channel_dict[self.signal]

        self.data_chunk = chunks[data_mask & channel_mask].iloc[0]
        self.param_chunks = [
            chunks[param_mask & channel_mask].iloc[0],
            chunks[acquisition_mask].iloc[0],
            chunks[optics_mask].iloc[0],
            chunks[sample_mask].iloc[0]
        ]

    def _parse_param_block(self, offset, length, param_dict):
        param_bin = self._bin_data[offset:offset + length * 4]
        i = 0

        while i < len(param_bin):
            tag = param_bin[i:i + 3].decode('utf-8')
            if tag == 'END':
                break
            i += 4
            dtype = struct.unpack('<H', param_bin[i:i + 2])[0]
            byte_length = struct.unpack('<H', param_bin[i + 2:i + 4])[0] * 2
            i += 4
            if dtype >= 2:
                content = param_bin[i:i + byte_length].rstrip(b'\x00').decode('utf-8')
            else:
                content = struct.unpack(self.type_dict[dtype, byte_length], param_bin[i:i + byte_length])[0]
            param_dict[tag] = content
            i += byte_length

        return param_dict

    def _parse_param_blocks(self):
        opus_logger.debug('Reading analysis parameters')
        params_tmp = {}
        for i, block in enumerate(self.param_chunks):
            opus_logger.debug(f'Reading parameter block {i}')
            params_tmp = self._parse_param_block(block.offset, block.length, params_tmp)

        self.params.append(params_tmp)

    def _parse_data_block(self):
        opus_logger.debug('Reading data block')
        offset = self.data_chunk.offset
        length = self.data_chunk.length
        data_bin = self._bin_data[offset:offset + length * 4]

        if not self.params:
            raise ValueError('Parameter list is empty. Was \'_parse_param_blocks\' executed first?')

        if len(data_bin) > (self.params[-1]['NPT'] + 1) * 4:
            opus_logger.debug('Found multiple spectra in file')
            data_tmp = self._parse_data_multiple(data_bin)
        else:
            opus_logger.debug('Found single spectrum in file')
            data_tmp = self._parse_data_single(data_bin).reshape(1, -1)

        self.data.append(data_tmp)

    def _parse_data_single(self, data_bin):
        npt = self.params[-1]['NPT']
        if len(data_bin) > npt * 4:
            data_bin = data_bin[:4 * npt]
        return np.asarray(struct.unpack('<' + 'f' * npt, data_bin))

    def _parse_data_multiple(self, data_bin):
        header = struct.unpack('<' + 'I' * 4, data_bin[4:20])

        data = []
        ix = header[1]
        i = 0

        while i < header[0]:
            tmp = data_bin[ix:ix + header[2]]
            data.append(self._parse_data_single(tmp))
            ix += header[2] + header[3]
            i += 1
        return np.stack(data)

    def _clean_data(self):
        opus_logger.debug('Cleaning up parsed data')
        self.params = pd.DataFrame(self.params)
        reps = [len(array) for array in self.data]
        self.params = self.params.loc[self.params.index.repeat(reps)]
        files = [file.relative_to(self._basepath) for file in self.files]

        # separate filename and parent directories
        self.params['parent'] = np.repeat([file.parent for file in files], reps)
        self.params['orig_file'] = np.repeat([file.name for file in files], reps)
        self.params['spectrum_no'] = np.concatenate([np.arange(n) for n in reps])

        # index = pd.MultiIndex.from_arrays([np.repeat(parents, reps),
        #                                    np.repeat(self.files, reps),
        #                                    np.concatenate([np.arange(n) for n in reps])],
        #                                   names=['parent', 'orig_file', 'spectrum_no'])
        # self.params.index = index

        # Calculate wavenumbers and add to data
        wn_params = self.params.loc[:, ['NPT', 'FXV', 'LXV']].to_records(index=False)
        if np.any(wn_params != wn_params[0]):
            raise ValueError('One or more files use a different spectral range.')
        wn_params = wn_params[0]

        wns = np.linspace(wn_params[1], wn_params[2], wn_params[0])
        self.data = pd.DataFrame(np.row_stack(self.data), columns=wns)

        # Collect metadata
        self.metadata = self.params.loc[:, ['parent', 'orig_file', 'spectrum_no',
                                            'DAT', 'SNM', 'SFM', 'SRC', 'RLP',
                                            'GRN', 'APT', 'INT', 'ASS']]
        self.metadata = self._format_metadata(self.metadata)

    @staticmethod
    def clean_string(s):
        s = re.sub(r'[^\w\s_]', '', s)

        s = re.sub(r'[\s._\-]+', '_', s)

        return s

    def _format_metadata(self, metadata):
        opus_logger.debug('Applying metadata format')
        metadata.columns = ['parent', 'orig_file', 'spectrum_no', 'date',
                            'sample_name', 'sample_form', 'laser', 'power',
                            'grating', 'aperture', 'integration_time', 'co_additions']

        metadata.date = pd.to_datetime(metadata.date)
        metadata.sample_name = [self.clean_string(s) for s in metadata.sample_name]
        metadata.sample_form = [self.clean_string(s) for s in metadata.sample_form]
        metadata.laser = [re.sub(r'\s', '', s) for s in metadata.laser]
        metadata.grating = [re.search(r', (\d+[a-z]),', s).group(1) for s in metadata.grating]
        metadata.aperture = [re.sub(r'\s', '', s) for s in metadata.aperture]

        return metadata

    @classmethod
    def from_dir(cls, path, signal='raman', metadata=False, recursive=False):
        opus_logger.info('Constructing OpusParser from directory path')
        opus_logger.debug(f'Received path: {path}')
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f'{str(path)} does not exist')
        if not path.is_dir():
            raise NotADirectoryError(f'{str(path)} is not a directory')

        opus_re = re.compile(r'.*\.\d+$')
        if recursive:
            opus_logger.debug('Collecting Opus files recursively')
            files = [file for file in path.rglob('*.*[0-9]') if opus_re.match(str(file))]
            # dirs = [file.parent for file in files]
        else:
            opus_logger.debug('Collecting Opus files')
            files = [file for file in path.glob('*.*[0-9]') if opus_re.match(str(file))]
        return cls(files, signal=signal, metadata=metadata, _basepath=path)

    def parse(self):
        opus_logger.info('Beginning parsing...')
        self._validate_params()

        for file in self.files:
            opus_logger.info(f'Reading file: {file}')
            with open(file, 'rb') as f:
                self._bin_data = f.read()
            chunks = self._parse_header()
            self._create_masks(chunks)
            self._parse_param_blocks()
            self._parse_data_block()
        self._clean_data()

    def export_data(self, path, single=True, **kwargs):
        opus_logger.info(f'Writing data to {path}')
        path = Path(path)

        if path.exists() and not path.is_dir():
            raise NotADirectoryError(
                f'Path is expected to be a directory when `single=True`, received {str(path)} instead')

        parent: Path
        for parent, meta_tmp in self.metadata.groupby('parent'):
            opus_logger.info(f'Writing {len(meta_tmp)} spectra to subdirectory {parent}')
            folder = path / parent
            folder.mkdir(parents=True, exist_ok=True)

            ix = np.asarray(self.metadata.parent == parent)
            data_tmp = self.data.iloc[ix]

            filename_counts = Counter()
            filenames_out = []
            total_files = len(data_tmp)

            if single:
                for i, row in enumerate(meta_tmp.itertuples()):
                    opus_logger.debug(f'Exporting spectrum {i} of {total_files}')
                    filename = '_'.join([row.date.strftime('%y%m%d'), row.sample_name, row.sample_form])
                    filename_full = filename + f'_{filename_counts[filename.lower()]:03}.csv'
                    filenames_out.append(filename_full)
                    filename_counts[filename.lower()] += 1

                    data_tmp.iloc[i].to_csv(path / row.parent / filename_full, header=False, **kwargs)

                if self.store_metadata:
                    opus_logger.debug('Exporting metadata')
                    # noinspection PyTypeChecker
                    meta_tmp.insert(0, 'file', filenames_out)
                    # self.metadata.reset_index(inplace=True)
                    meta_tmp.drop(columns='spectrum_no', inplace=True)
                    meta_tmp.set_index('file', inplace=True)
                    meta_tmp.to_csv(path / parent / 'metadata.csv')
        # else:
        #    for row1, row2 in zip(self.data.iterrows(), self.metadata.itertuples()):


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

    parser = OpusParser.from_dir('/home/daniel/data_ecoli/', metadata=True, recursive=True)
    parser.parse()
    parser.export_data('/home/daniel/data_ecoli_out')
