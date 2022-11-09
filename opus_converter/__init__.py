import re
import struct

import numpy as np 

def read_opus(path, metadata=False):
    with open(path, "rb") as f:
        data = f.read()
    
    opus_regex = re.compile(br"""
        \x00{5}NPT\x00{3}\x02\x00(.{4})                 # Number of points
        FXV\x00\x01\x00\x04\x00(.{8})                   # First wavenumber
        LXV\x00\x01\x00\x04\x00(.{8})                   # Last wavenumber
        CSF\x00.{12}MXY\x00.{12}                        # Not used
        MNY\x00.{12}DPF\x00.{8}                         # Not used
        DAT\x00.{4}(.{10})\x00\x00                      # Date
        TIM\x00.{4}(.{16}).{4}                          # Time
        DXU\x00.{8}                                     # Not used
        END\x00{9}(.*?)\x00{4}NPT                       # Raman Data
    """, re.VERBOSE | re.DOTALL)

    match = opus_regex.search(data)
    npt, fxv, lxv, dat, tim, ints = match.groups()

    npt = struct.unpack("<I", npt)[0]
    fxv = struct.unpack("<d", fxv)[0]
    lxv = struct.unpack("<d", lxv)[0]
    dat = dat.decode("ascii")
    tim = tim.decode("ascii")

    wns = np.linspace(fxv, lxv, npt)
    ints = np.asarray(struct.unpack("<" + "f"*npt, ints))

    data_out = np.column_stack((wns, ints))

    # TODO: Handle metadata
    mdata = {
        "date": dat,
        "time": tim
    }

    if metadata:
        return data_out, mdata
    else:
        return data_out
