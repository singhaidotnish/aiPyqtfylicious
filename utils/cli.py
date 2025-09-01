# sub commands
'''
import yaml
from pathlib import Path
import typer
import importlib.util as iu

app = typer.Typer()
ROOT = Path(__file__).resolve().parents[1]  # project root


def load_module_from_path(path: Path):
    spec = iu.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@app.command()
def run(
    exp: str = typer.Argument(..., help="Experiment filename without .py, e.g. 02_head_ablation"),
    cfg: str = typer.Option("experiments/config.yaml", help="Path to config file"),
):
    cfg_path = (ROOT / cfg).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg_data = yaml.safe_load(cfg_path.read_text())

    exp_path = (ROOT / "experiments" / f"{exp}.py").resolve()
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment not found: {exp_path}")

    mod = load_module_from_path(exp_path)
    if not hasattr(mod, "main"):
        raise AttributeError(f"{exp_path.name} must define a function main(cfg)")

    mod.main(cfg_data)


if __name__ == "__main__":
    app()
'''


# single command


# utils/cli.py
import yaml
from pathlib import Path
import typer
import importlib.util as iu

ROOT = Path(__file__).resolve().parents[1]

def load_module_from_path(path: Path):
    spec = iu.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main(
    exp: str = typer.Argument(..., help="Experiment file without .py, e.g. 02_head_ablation"),
    cfg: str = typer.Option("experiments/config.yaml", help="Path to config"),
):
    cfg_path = (ROOT / cfg)
    cfg_data = yaml.safe_load(cfg_path.read_text())

    exp_path = (ROOT / "experiments" / f"{exp}.py")
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment not found: {exp_path}")
    mod = load_module_from_path(exp_path)
    if not hasattr(mod, "main"):
        raise AttributeError(f"{exp_path.name} must define main(cfg)")
    mod.main(cfg_data)

if __name__ == "__main__":
    typer.run(main)
