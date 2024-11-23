import argparse
import builtins
from contextlib import nullcontext
import sys
from types import ModuleType
from typing import Any

import pkg_resources


class DummyModule(ModuleType):
    def __getattr__(self, key):
        return DummyModule(name=f"{self.__name__}.{key}")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self

    def __mro_entries__(self, bases):
        return (self, self)

    def __init__(self, name: str, doc: str | None = "", *rest) -> None:
        if type(name) is str and (doc is None or type(doc) is str):
            super().__init__(name, doc)

    __all__ = []  # support wildcard imports


class DummyImport:
    def __init__(self, dummy_pkgs: list[str], import_err: str) -> None:
        self.dummy_pkgs = [pkg.replace("-", "_") for pkg in dummy_pkgs]
        self.import_err = import_err
        self.realimport = builtins.__import__

    def __enter__(self):
        def tryimport(name, globals={}, locals={}, fromlist=[], level=0):
            pkg_name = name.split(".")[0].replace("-", "_")
            if pkg_name in self.dummy_pkgs:
                return DummyModule(name=name)
            try:
                return self.realimport(name, globals, locals, fromlist, level)
            except ImportError as err:
                print("Tried to import", name, file=sys.stderr)
                print(self.import_err, file=sys.stderr)
                print("Error", err, file=sys.stderr)
                raise err

        builtins.__import__ = tryimport

    def __exit__(self, exc_type, exc_value, traceback):
        builtins.__import__ = self.realimport


# orchestrate all parsers
def parse_intermixed_args(uninstalled_requirements=False) -> argparse.Namespace:
    requirements_path = "./requirements.txt"
    with open(requirements_path, "r") as f:
        requirements = [req.project_name for req in pkg_resources.parse_requirements(f)]

    parser = argparse.ArgumentParser(
        add_help=False,
        prog="ail_run.sh",
        description="Run commands on the AI-Lab frontend.",
    )
    parser.add_argument(
        "--no_sync",
        "-N",
        action="store_true",
        help="LOCAL: Do not sync the codebase before and after running the job.",
    )
    parser.add_argument(
        "--keep_jobs",
        "-k",
        action="store_true",
        help="FE: Do not cancel previous jobs before running the new job.",
    )
    parser.add_argument(
        "--scmds_from",
        "-s",
        type=str,
        default="ail_fe_main_scmds",
        help="FE: The Python file to import SCmds from.",
    )
    parser.add_argument(
        "--no_install",
        "-n",
        action="store_true",
        help="SLURM: Do not install packages from requirements.txt before running the Python command.",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,
        help="SLURM: The Python file to run in the Slurm job with torchrun.",
    )
    (args, rest) = parser.parse_known_intermixed_args()

    with (
        DummyImport(
            dummy_pkgs=requirements,
            import_err="Make sure to add the package to requirements.txt!",
        )
        if uninstalled_requirements
        else nullcontext()
    ):
        try:
            scmds_module = __import__(args.scmds_from)
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                f"Incorrect --scmds_from: Module {args.scmds_from} does not exist."
            )
        if not hasattr(scmds_module, "get_scmds"):
            raise AttributeError(
                f"Module {args.scmds_from} does not have a get_scmds function."
            )
        if not hasattr(scmds_module, "get_parser"):
            raise AttributeError(
                f"Module {args.scmds_from} does not have a get_parser function."
            )
        parser = scmds_module.get_parser(parser)

        try:
            py_slurm_module = __import__(args.file)
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                f"Incorrect -f: Module {args.file} does not exist."
            )
        if not hasattr(py_slurm_module, "get_parser"):
            raise AttributeError(
                f"Module {args.file} does not have a get_parser function."
            )
        parser = py_slurm_module.get_parser(parser)

    parser.add_argument(
        "--help",
        "-h",
        action="help",
        help="Show this help message and exit.",
    )

    return parser.parse_intermixed_args()
