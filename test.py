import os
from opus_converter import read_opus

files = sorted(os.listdir("./data"))
for file in files:
    print(file)
    try:
        data, metadata = read_opus("./data/" + file, metadata=True)
    except AttributeError:
        print("Parsing failed")    
    else:
        print(f"Loaded OPUS file with {data.shape} wavenumber points ranging from {data[0,0]} to {data[-1, 0]}.")
        for key, val in metadata.items():
            print(f"{key}: {val}")