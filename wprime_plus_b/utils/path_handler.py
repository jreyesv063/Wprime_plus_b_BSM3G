import os
import pathlib


class Paths:

    def __init__(self) -> None:
        """Initializes Paths instance.

        Finds the root path as the directory two levels upwards of where this file is located.
        Prints out the detected root path.
        """
        self.root_path = pathlib.Path(__file__).resolve().parent.parent

    @staticmethod
    def safe_return(path: pathlib.Path, path_type: str, mkdir: bool) -> pathlib.Path:
        """Safely return a path by optionally creating the parent directories to avoid errors when writing to the path.

        Args:
            path: Path to optionally create and return.
            path_type: Whether the path points to a directory or file
                (relevant for creating the path or parent path respectively).
            mkdir: If True, creates the parent directories. If False, it has no effect.

        Returns:
            Input path.
        """
        if mkdir:
            if path_type == "file":
                path.parent.mkdir(parents=True, exist_ok=True)
            elif path_type == "directory":
                path.mkdir(parents=True, exist_ok=True)
            else:
                raise ValueError(
                    f"'path_type' has to be either 'file' or 'directory'. "
                    f"Got: {path_type}"
                )
        return path

    def processor_path(
        self,
        processor_name: str,
        processor_lepton_flavour: str = None,
        processor_channel: str = None,
        dataset_year: str = None,
        mkdir: bool = None,
    ):
        processor_path = "/".join(
            [
                elem
                for elem in [
                    processor_name,
                    processor_channel,
                    processor_lepton_flavour,
                    dataset_year,
                ]
                if elem is not None
            ]
        )
        # make output directory
        output_path = self.safe_return(
            path=self.root_path / "outs" / processor_path,
            path_type="directory",
            mkdir=mkdir,
        )
        # make output metadata directory 
        self.safe_return(
            path=output_path / "metadata",
            path_type="directory",
            mkdir=mkdir,
        )
        return output_path