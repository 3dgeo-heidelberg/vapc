from pathlib import Path
import os
import json
import pytest
import subprocess


CMD_TEMPLATES_FOLDER = "command_line_templates"
WORKING_DIR = Path(__file__).parent.parent.parent


def run_template(config_file, temp_folder):
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
    result_folder = run_template(config_file, tmp_path)
    assert Path(result_folder).exists()
    assert len(list(Path(result_folder).rglob("*"))) > 0
