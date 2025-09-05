from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import box
import sys, os, datetime

console = Console()

def get_progress():
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        transient=True
    )

def log_file():
    log_path = os.path.join(os.getenv("ARTIFACTS_DIR", "artifacts"), "run.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    return open(log_path, "a", encoding="utf-8", buffering=1)

def info(msg: str):
    console.print(f"[bold cyan]{msg}[/bold cyan]")
    print(msg, file=log_file())

def warn(msg: str):
    console.print(f"[bold yellow]WARN:[/bold yellow] {msg}")
    print(f"WARN: {msg}", file=log_file())

def error(msg: str):
    console.print(f"[bold red]ERR:[/bold red] {msg}")
    print(f"ERR: {msg}", file=log_file())
