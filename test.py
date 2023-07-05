import os
from opus_converter import OpusParser

files = sorted(os.listdir("./data"))

print(files)
try:
    parser = OpusParser.from_dir("./data/", metadata=True)
    parser.parse()
except AttributeError:
    print("Parsing failed")
else:
    print(f"Loaded {len(parser.metadata)} OPUS files with {parser.data.shape[-1]} wavenumber points, "
          f"ranging from {parser.data.columns[0]} to {parser.data.columns[-1]}.")
    print("Metadata:")
    print(parser.metadata.columns)
