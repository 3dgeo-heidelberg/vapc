from pathlib import Path
import os
import json
import pytest
import subprocess


CMD_TEMPLATES_FOLDER = "command_line_templates"
WORKING_DIR = Path(__file__).parent.parent.parent


def run_template(config_file, temp_folder):
    """
    Function to run the command line templates (json files)
    in a specific folder and write the output to a temporary folder.

    Parameters
    ----------
    config_file : str
        Path to the json file containing the command line template.
    temp_folder : str

    Returns
    -------
    str
        Path to the output folder.
    """
    outfolder = str(temp_folder / "output")
    Path(outfolder).mkdir(parents=True, exist_ok=True)
    with open(config_file, "r") as file:
        config = json.load(file)
        config["outdir"] = outfolder
    new_config_file = temp_folder / Path(config_file).name
    with open(new_config_file, "w") as outfile:
        json.dump(config, outfile)
    subprocess.run(
        ["python", "run.py", str(new_config_file)], cwd=WORKING_DIR, check=True
    )
    return outfolder

@pytest.mark.cmdtest
@pytest.mark.parametrize("config_file", 
                         list(Path(CMD_TEMPLATES_FOLDER).rglob("*.json"))
)
def test_template(config_file, tmp_path):
    """
    Test running the command line templates in the CMD_TEMPLATES_FOLDER.
    """
    # run command
    result_folder = run_template(config_file, tmp_path)
    # check if the output folder exists and is not empty
    assert Path(result_folder).exists()
    assert len(list(Path(result_folder).rglob("*"))) > 0
