from opus_parser.opus_parser import OpusHeader

with open("data/test_file_2_1.0", "rb") as f:
    data = f.read()

header = OpusHeader(data[:504])
header.parse()

for block in header.blocks:
    print("Offset:", block.offset)
    print("Length:", block.length * 4)
    print("Type:", block.type)
    print("Channel:", block.channel)
    print("Text:", block.text)
    print("=====================================")
