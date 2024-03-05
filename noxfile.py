"""Task runner for the developer

# Usage

```
nox -l            # list of sessions.
nox -s <session>  # execute a session
nox -k <keyword>  # execute some session
```

"""

import os

from datetime import timedelta
from functools import partial
from pathlib import Path
from time import time

import nox

TEST_ENV_VARS = {"OMP_NUM_THREADS": "1"}


no_venv_session = partial(nox.session, venv_backend="none")
os.environ.update({"PDM_IGNORE_SAVED_PYTHON": "1"})
nox.options.sessions = ["test(cov=True, with_opencv=True)"]
nox.options.reuse_existing_virtualenvs = 1


@nox.session
def validate_code(session):
    session.run_always(
        "pdm", "sync", "--clean", "-G", "dev", "--no-self", external=True
    )
    session.run("pdm", "validate_code", external=True)


class TimePrinter:
    def __init__(self):
        self.time_start = self.time_last = time()

    def __call__(self, task: str):
        time_now = time()
        if self.time_start != self.time_last:
            print(
                f"Time for {task}: {timedelta(seconds=time_now - self.time_last)}"
            )
        print(
            f"Session started since {timedelta(seconds=time_now - self.time_start)}"
        )
        self.time_last = time_now


@nox.parametrize("with_opencv", [True, False])
@nox.parametrize("cov", [True, False])
@nox.session
def test(session, cov, with_opencv):
    """Execute unit-tests using pytest"""

    command = "pdm sync --clean -G test --no-self"
    if with_opencv:
        command += " -G opencv"

    session.run_always(*command.split(), external=True)
    session.install(".", "--no-deps")

    args_cov = []
    if cov:
        args_cov.extend([
            "--cov",
            "--no-cov-on-fail",
            "--cov-report=term-missing",
        ])

    session.run(
        "python",
        "-m",
        "pytest",
        "--pyargs",
        "fluidimage",
        *args_cov,
        *session.posargs,
        env=TEST_ENV_VARS,
    )
    if cov:
        session.run("coverage", "xml")


@nox.session(name="test-examples")
def test_examples(session):
    """Execute the examples using pytest"""

    command = "pdm sync --clean -G test --no-self"
    session.run_always(*command.split(), external=True)

    session.install(".", "--no-deps")

    session.chdir("doc/examples")
    session.run("pytest", "-v")


@nox.session
def doc(session):
    """Build the documentation"""
    print_times = TimePrinter()
    command = "pdm sync -G doc -G opencv --no-self"
    session.run_always(*command.split(), external=True)
    print_times("pdm sync")

    session.install(
        ".", "-C", "setup-args=-Dtransonic-backend=python", "--no-deps"
    )
    print_times("install self")

    session.chdir("doc")
    session.run("make", "cleanall", external=True)
    session.run("make", external=True)
    print_times("make doc")


def _get_version_from_pyproject(path=Path.cwd()):
    if isinstance(path, str):
        path = Path(path)

    if path.name != "pyproject.toml":
        path /= "pyproject.toml"

    in_project = False
    version = None
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.startswith("[project]"):
                in_project = True
            if line.startswith("version =") and in_project:
                version = line.split("=")[1].strip()
                version = version[1:-1]
                break

    assert version is not None
    return version


@nox.session(name="add-tag-for-release", venv_backend="none")
def add_tag_for_release(session):
    """Add a tag to the repo for a new version"""
    session.run("hg", "pull", external=True)

    result = session.run(
        *"hg log -r default -G".split(), external=True, silent=True
    )
    if result[0] != "@":
        session.run("hg", "update", "default", external=True)

    version = _get_version_from_pyproject()
    print(f"{version = }")

    result = session.run("hg", "tags", "-T", "{tag},", external=True, silent=True)
    last_tag = result.split(",", 2)[1]
    print(f"{last_tag = }")

    if last_tag == version:
        session.error("last_tag == version")

    answer = input(
        f'Do you really want to add and push the new tag "{version}"? (yes/[no]) '
    )

    if answer != "yes":
        print("Maybe next time then. Bye!")
        return

    print("Let's go!")
    session.run("hg", "tag", version, external=True)
    session.run("hg", "push", external=True)
