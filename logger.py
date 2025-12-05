import sys
import time
import uuid
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from loguru import logger


def tqdm_sink(message: str) -> None:
    tqdm.write(message.rstrip("\n"))


class Logger:
    """
    Logging all details about the current run state,
    details about the current state of the application,
    and helper methods for logging progress, sections, rules, etc.
    """
    _crash_hook_installed: bool = False  

    def __init__(
        self,
        app: str = "WSOD-YOLO",
        level: str = "INFO",
        log_dir: str | Path = "logs",
        serialize: bool = False,
        rich_tracebacks: bool = True,
        use_tqdm_sink: bool = True,
    ) -> None:
        assert app, "App name must be provided"
        assert log_dir, "Log directory must be provided"
        assert level, "Log level must be provided"

        self.app: str = app
        self.run_id: str = uuid.uuid4().hex[:8].upper()
        self.log_dir: Path = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.formatter: str = (
            "<green>{time:HH:mm:ss}</green> | "
            f"<cyan>{self.app}</cyan>:<m>{self.run_id}</m> | "
            "<lvl>{level: <7}</lvl> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
        self.serialize: bool = serialize
        self.rich_tracebacks: bool = rich_tracebacks
        self.use_tqdm_sink: bool = use_tqdm_sink
        self.level: str = level

        logger.remove()
        sink = tqdm_sink if use_tqdm_sink else sys.stdout
        logger.add(
            sink,
            level=level,
            colorize=True,
            format=self.formatter,
            enqueue=True,
            backtrace=rich_tracebacks,
            diagnose=False,
        )

        day = datetime.now().strftime("%Y-%m-%d")
        self.main_file: Path = self.log_dir / f"{self.app.lower()}_{day}.log"
        self.err_file: Path = self.log_dir / f"{self.app.lower()}_{day}_errors.log"
        self.warnings_file: Path = self.log_dir / f"{self.app.lower()}_{day}_warnings.log"
        logger.add(
            self.main_file,
            level="DEBUG",
            format=self.formatter,
            enqueue=True,
            rotation="1 day",
            retention="1 day",
            compression="zip",
            serialize=serialize,
        )
        logger.add(
            self.err_file,
            level="ERROR",
            format=self.formatter,
            enqueue=True,
            rotation="1 day",
            retention="1 day",
            compression="zip",
            serialize=serialize,
        )
        logger.add(
            self.warnings_file,
            level="WARNING",
            format=self.formatter,
            enqueue=True,
            rotation="1 day",
            retention="1 day",
            compression="zip",
            serialize=serialize,
        )

        self.logger = logger.bind(app=self.app, run_id=self.run_id)
        self.started: float | None = None
        # self.excepthook = sys.excepthook
        # self.install_crash_cleaner()

    @property
    def log(self):
        return self.logger

    def info(self, *args, **kwargs):
        return self.logger.info(*args, **kwargs)

    def debug(self, *args, **kwargs):
        return self.logger.debug(*args, **kwargs)

    def warning(self, *args, **kwargs):
        return self.logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        return self.logger.error(*args, **kwargs)

    def install_crash_cleaner(self) -> None:
        """
        Install a global exception hook that clears the log files when
        an unhandled exception occurs.
        """
        if Logger._crash_hook_installed:
            return
        Logger._crash_hook_installed = True

        prev_hook = sys.excepthook
        self.excepthook = prev_hook

        def handle_exception(exc_type: type, exc: BaseException, exc_tb) -> None:
            if exc_type is KeyboardInterrupt:
                return prev_hook(exc_type, exc, exc_tb)
            try:
                self.logger.error("Unhandled exception, clearing log files...")
            except Exception:
                pass
            self.clear_log_files()
            prev_hook(exc_type, exc, exc_tb)
        sys.excepthook = handle_exception

    def clear_log_files(self) -> None:
        """(Optional) Mark crash in logs instead of truncating them."""
        try:
            with open(self.main_file, "a", encoding="utf-8") as f:
                f.write("\n===== CRASH DETECTED =====\n")
            with open(self.err_file, "a", encoding="utf-8") as f:
                f.write("\n===== CRASH DETECTED =====\n")
        except OSError:
            pass

    def set_level(self, level: str = "INFO") -> None:
        self.level = level
        logger.remove()
        sink = tqdm_sink if self.use_tqdm_sink else sys.stdout
        logger.add(
            sink,
            level=level,
            colorize=True,
            format=self.formatter,
            enqueue=True,
            backtrace=self.rich_tracebacks,
            diagnose=False,
        )

        logger.add(
            self.main_file,
            level="DEBUG",
            format=self.formatter,
            enqueue=True,
            rotation="1 day",
            retention="1 day",
            compression="zip",
            serialize=self.serialize,
        )
        logger.add(
            self.err_file,
            level="ERROR",
            format=self.formatter,
            enqueue=True,
            rotation="1 day",
            retention="1 day",
            compression="zip",
            serialize=self.serialize,
        )
        self.logger = logger.bind(app=self.app, run_id=self.run_id)

    def add_file(
        self,
        path: str | Path,
        level: str = "INFO",
        rotation: str = "100 MB",
        retention: str = "1 day",
    ) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            p,
            level=level,
            format=self.formatter,
            enqueue=True,
            rotation=rotation,
            retention=retention,
            compression="zip",
            serialize=self.serialize,
        )

    def bind(self, **extra) -> "Logger":
        return self.__class__.from_logger(
            logger.bind(**extra),
            self.app,
            self.run_id,
            self.log_dir,
            self.formatter,
            self.serialize,
            self.rich_tracebacks,
            self.use_tqdm_sink,
            self.level,
            self.main_file,
            self.err_file,
            self.warnings_file,
        )

    @classmethod
    def from_logger(
        cls,
        logger_obj,
        app: str,
        run_id: str,
        log_dir: Path,
        formatter: str,
        serialize: bool,
        rich_tracebacks: bool,
        use_tqdm_sink: bool,
        level: str,
        main_file: Path,
        err_file: Path,
        warnings_file: Path,
    ) -> "Logger":
        obj = object.__new__(cls)
        # FIELDS for paths and identification components
        obj.app, obj.run_id, obj.log_dir = app, run_id, Path(log_dir)

        # COMMON handlers for both console and file
        obj.formatter, obj.serialize, obj.rich_tracebacks = (
            formatter,
            serialize,
            rich_tracebacks,
        )
        obj.use_tqdm_sink, obj.level = use_tqdm_sink, level

        # FILE handlers
        obj.main_file, obj.err_file, obj.warnings_file = main_file, err_file, warnings_file
        obj.logger, obj.started = logger_obj, None
        return obj

    def section(self, title: str, level: str = "INFO", char: str = "=") -> None:
        line = char * max(8, len(title) + 6)
        self.logger.log(level, "")
        self.logger.log(level, line)
        self.logger.log(level, title)
        self.logger.log(level, line)

    def rule(self, level: str = "INFO", char: str = "-", width: int = 56) -> None:
        self.logger.log(level, char * width)

    def start_task(self, name: str, level: str = "INFO") -> None:
        self.started = time.perf_counter()
        self.logger.log(level, f"{name} | start")

    def end_task(self, name: str, level: str = "SUCCESS") -> None:
        t1 = time.perf_counter()
        dt = (t1 - (self.started or t1)) * 1000
        self.logger.log(level, f"{name} | done in {dt:.1f} ms")

    def progress(self, total: int, desc: str = "", leave: bool = True) -> tqdm:
        return tqdm(total=total, desc=desc, leave=leave, dynamic_ncols=True, mininterval=0.2)
