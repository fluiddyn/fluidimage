"""Task runner for the developer

Usage
-----

   nox -l

   nox -s <session>

   nox -k <keyword>
or:

   make <session>

execute ``make list-sessions```` or ``nox -l`` for a list of sessions.

"""
import re
import shlex
from pathlib import Path
from functools import partial

import nox

PACKAGE = "snek5000"
CWD = Path.cwd()

if (CWD / "poetry.lock").exists():
    BUILD_SYSTEM = "poetry"
    PACKAGE_SPEC = "pyproject.toml"
else:
    BUILD_SYSTEM = "setuptools"
    PACKAGE_SPEC = "setup.cfg"

TEST_ENV_VARS = {"OMP_NUM_THREADS": "1"}

EXTRA_REQUIRES = ("main", "doc", "test", "dev")


@nox.session(name="pip-compile", reuse_venv=True)
@nox.parametrize(
    "extra", [nox.param(extra, id=extra) for extra in EXTRA_REQUIRES]
)
def pip_compile(session, extra):
    """Pin dependencies to requirements/*.txt

    How to run all::

        pipx install nox
        nox -s pip-compile

    """
    session.install("pip-tools")
    req = Path("requirements")

    if extra == "main":
        in_extra = ""
        in_file = ""
    else:
        in_extra = f"--extra {extra}"
        in_file = req / "vcs_packages.in"

    out_file = req / f"{extra}.txt"

    session.run(
        *shlex.split(
            "python -m piptools compile --resolver backtracking --quiet "
            f"{in_extra} {in_file} {PACKAGE_SPEC} "
            f"-o {out_file}"
        ),
        *session.posargs,
    )

    session.log(f"Removing absolute paths from {out_file}")
    packages = out_file.read_text()
    rel_path_packages = packages.replace(
        "file://" + str(Path.cwd().resolve()), "."
    )

    if extra == "tests":
        tests_editable = out_file.parent / out_file.name.replace(
            "tests", "tests-editable"
        )
        session.log(f"Copying {out_file} with -e flag in {tests_editable}")
        tests_editable.write_text(rel_path_packages)
        session.log(f"Removing -e flag in {out_file}")
        rel_path_packages = re.sub(r"^-e\ \.", ".", rel_path_packages, flags=re.M)

    session.log(f"Writing {out_file}")
    out_file.write_text(rel_path_packages)


no_venv_session = partial(nox.session, venv_backend="none")
nox.options.sessions = ["tests"]


@nox.session
def tests(session):
    """Execute unit-tests using pytest"""

    session.install("-r", "requirements/test.txt")
    session.run(
        "python",
        "-m",
        "pytest",
        *session.posargs,
        env=TEST_ENV_VARS,
    )


@no_venv_session(name="tests-cov")
def tests_cov(session):
    """Execute unit-tests using pytest+pytest-cov"""
    session.notify(
        "tests",
        [
            "--cov",
            # "--cov-config=setup.cfg",
            "--no-cov-on-fail",
            "--cov-report=term-missing",
            *session.posargs,
        ],
    )


@nox.session(name="coverage-html")
def coverage_html(session, nox=False):
    """Generate coverage report in HTML. Requires `tests-cov` session."""
    report = Path.cwd() / ".coverage" / "html" / "index.html"
    session.install("coverage[toml]")
    session.run("coverage", "html")

    print("Code coverage analysis complete. View detailed report:")
    print(f"file://{report}")
