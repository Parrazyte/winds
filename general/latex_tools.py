import os
from pathlib import Path
import shutil
import time

def merge_paper_file_dir():
    '''
    Stolen from the dark side.
    Copies all files in subdirectories into the current working directory
    Useful for AAS submissions
    '''


    DELAY_SECONDS = 0.1  # Adjust as needed

    cwd = Path.cwd()

    for path in cwd.rglob("*"):
        if path.is_file() and path.parent != cwd:
            destination = cwd / path.name

            # Handle filename collisions
            if destination.exists():
                stem = destination.stem
                suffix = destination.suffix
                counter = 1

                while destination.exists():
                    destination = cwd / f"{stem}_{counter}{suffix}"
                    counter += 1

            print(f"Moving: {path} -> {destination}")
            shutil.copy(str(path), str(destination))

            # Small delay after each file
            time.sleep(DELAY_SECONDS)

    print("Done.")

def remove_dirs_latex_includegraphics(file):
    '''
    Stolen from the dark side
    removes all directories listed within an includegraphics in a latex file
    Useful for AAS submissions
    '''
    import os
    from pathlib import Path
    assert file.endswith('.tex'),'Error: non-tex file detected'

    file_onedir =file.replace('.tex','_onedir.tex')
    os.system('cp '+file+' '+file_onedir)
    latex_file_onedir = Path(file_onedir)

    pattern = re.compile(r'(\\includegraphics(?:\[[^\]]*\])?\{)([^}]+)(\})')

    with open(latex_file_onedir, "r", encoding="utf-8") as f:
        content = f.read()

    def replace_path(match):
        prefix, path, suffix = match.groups()
        filename = Path(path).name
        return f"{prefix}{filename}{suffix}"

    new_content = pattern.sub(replace_path, content)

    with open(latex_file_onedir, "w", encoding="utf-8") as f:
        f.write(new_content)

    print("Updated all \\includegraphics paths.")