from pathlib import Path


template = """
python_sources = [
{sources}
]

py.install_sources(
  python_sources,
  subdir: '{subdir}'
)

"""


def process_directory(path_dir):

    print(f"Process {path_dir}")

    names_py = sorted(path.name for path in path_dir.glob("*.py"))
    print(f"{names_py = }")
    paths_subdirs = [
        path
        for path in path_dir.glob("*")
        if path.is_dir() and not path.name.startswith("__")
    ]

    names_subdirs = sorted(path.name for path in paths_subdirs)
    print(f"{names_subdirs = }")

    code = template.format(sources="\n".join(f"  '{name}'," for name in names_py), subdir=path_dir)
    code += "\n".join(f"subdir('{name}')" for name in names_subdirs) + "\n"

    with open(path_dir / "meson.build", "w", encoding="utf-8") as file:
        file.write(code)

    for path_subdir in paths_subdirs:
        process_directory(path_subdir)


process_directory(Path("fluidimage"))
