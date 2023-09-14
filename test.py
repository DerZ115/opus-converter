import unittest
from opus_parser.parser import OpusFile, OpusHeader
from pathlib import Path


class TestOpusHeader(unittest.TestCase):
    def test_output_length(self):
        with open("test_files/test_raman_single.0", "rb") as f:
            header_to_test = f.read(504)
        header = OpusHeader(header_to_test)
        block_list = header.parse()
        self.assertEqual(len(block_list), 20)

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            OpusHeader(b"1234567890")

    def test_output_keys(self):
        with open("test_files/test_raman_single.0", "rb") as f:
            header_to_test = f.read(504)
        header = OpusHeader(header_to_test)
        block_list = header.parse()
        self.assertEqual(list(block_list[0].keys()),
                         ["offset",
                          "length",
                          "block_type"])


class TestFileValidation(unittest.TestCase):
    def test_validation_success(self):
        for file in Path("./test_files").glob("*.0"):
            with self.subTest(file=file):
                self.assertTrue(OpusFile(file)._validate_file())

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            OpusFile("./test_files/nonexistent.0")._validate_file()

    def test_directory(self):
        with self.assertRaises(IsADirectoryError):
            OpusFile("./test_files")._validate_file()

    def test_bad_file_format(self):
        with self.assertRaises(ValueError):
            OpusFile("./test_files/textfile.txt")._validate_file()


class TestOpusFile(unittest.TestCase):
    


if __name__ == '__main__':
    unittest.main()
