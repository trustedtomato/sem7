import argparse
import builtins
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
import logging
from os import devnull
import os
import sys
import traceback
from types import ModuleType
from typing import Any, Callable, Sequence

from ail_fe_main_scmds import SCmd


Parser = argparse.ArgumentParser | argparse._ArgumentGroup


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
                traceback.print_exception(err)
                raise err

        builtins.__import__ = tryimport

    def __exit__(self, exc_type, exc_value, traceback):
        builtins.__import__ = self.realimport


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


# Used by the target script. The unique thing about the target script is that it
# can be called from the frontend, or locally, on your machine. And due to
# limitations, the frontend also forwards all other arguments to the target
# script.
def parse_intermixed_args_local(
    modify_parser: Callable[[argparse.ArgumentParser | argparse._ArgumentGroup], None]
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    modify_parser(parser)

    with suppress_stdout_stderr():
        # first try with local-only arguments
        try:
            return parser.parse_intermixed_args()
        # Normally, we would use exit_on_error=False, but that doesn't work in
        # all cases in Python 3.10, which is what we have on the AI-Lab
        # frontend. See the issue here:
        # https://github.com/python/cpython/issues/103498
        except SystemExit:
            pass

        # now try with all frontend argument
        try:
            return parse_intermixed_args()
        except SystemExit:
            pass

    # if all of the above, raise the original error - we don't expect the call
    # to fail on the frontend because we have already tested it when running
    # ail_run.sh
    return parser.parse_intermixed_args()


# orchestrate all parsers
def parse_intermixed_args(
    uninstalled_requirements=False,
    sys_args: list[str] = sys.argv[1:],
) -> argparse.Namespace:
    requirements_path = "./requirements.txt"
    with open(requirements_path, "r") as f:
        requirements = [pkg.split("==")[0] for pkg in f.read().splitlines()]

    parser = argparse.ArgumentParser(
        add_help=False,
        prog="ail_run.sh",
        description="Run commands on the AI-Lab frontend.",
    )

    local_options = parser.add_argument_group("options for local")
    local_options.add_argument(
        "--no_sync",
        "-N",
        action="store_true",
        help="Do not sync the codebase before and after running the job.",
    )

    fe_options = parser.add_argument_group("options for the frontend")
    fe_options.add_argument(
        "--keep_jobs",
        "-k",
        action="store_true",
        help="Do not cancel previous jobs before running the new job.",
    )
    fe_options.add_argument(
        "--scmds_from",
        "-s",
        type=str,
        default="ail_fe_main_scmds",
        help="The Python file to import Slurm commands from.",
    )

    slurm_options = parser.add_argument_group("options for Slurm")
    slurm_options.add_argument(
        "--no_install",
        "-n",
        action="store_true",
        help="Do not install packages from requirements.txt before running the Python command.",
    )
    slurm_options.add_argument(
        "--torchrun",
        "-t",
        action="store_true",
        help="Use torchrun to run the Python command.",
    )

    (args, rest) = parser.parse_known_intermixed_args(sys_args)

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
        if not hasattr(scmds_module, "modify_parser"):
            raise AttributeError(
                f"Module {args.scmds_from} does not have a modify_parser function."
            )
        scmds_module.modify_parser(fe_options)

        (args, rest) = parser.parse_known_intermixed_args(sys_args)

        scmds: list[SCmd] = scmds_module.get_scmds(args)

        slurm_options.add_argument(
            "-f",
            "--file",
            type=str,
            required=any(scmd.python_module is None for scmd in scmds),
            help="The Python file to run in the Slurm job with torchrun.",
        )
        (args, rest) = parser.parse_known_intermixed_args(sys_args)

        for i, scmd in enumerate(scmds):
            py_slurm_module_name = scmd.python_module or args.file
            try:
                py_slurm_module = __import__(py_slurm_module_name)
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    f"Incorrect -f: Module {py_slurm_module_name} does not exist."
                )
            if not hasattr(py_slurm_module, "modify_parser"):
                raise AttributeError(
                    f"Module {py_slurm_module_name} does not have a modify_parser function."
                )
            if i == 0:
                target_options = parser.add_argument_group(
                    "options for the target script"
                )
                py_slurm_module.modify_parser(target_options)
                parser.add_argument(
                    "--help",
                    "-h",
                    action="help",
                    help="Show this help message and exit.",
                )
            parser.parse_intermixed_args(sys_args + scmd.python_args)

    return parser.parse_intermixed_args(sys_args + scmds[0].python_args)
