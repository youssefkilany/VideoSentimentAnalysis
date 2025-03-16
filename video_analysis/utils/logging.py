from dataclasses import dataclass


@dataclass
class LOG_LEVEL:
    ERROR_LOG = 0
    INFO_LOG = 1
    DEBUG_LOG = 2

    LOG_TITLE = {
        ERROR_LOG: "ERROR",
        INFO_LOG: "INFO",
        DEBUG_LOG: "DEBUG",
    }


@dataclass
class LOG_STREAM:
    STDOUT_LOG = 0
    FILE_LOG = 1
    STRING_LOG = 2


def log_msgs(
    *msgs,
    sep="\n",
    end="\n",
    log_level=LOG_LEVEL.INFO_LOG,
    log_stream=LOG_STREAM.STDOUT_LOG,
    log_file=None,
):
    # TODO: Add documentation

    # TODO: wrap these functions within a Logger class, and use a log-level variable that filters messages as per the logger instance used. usage should be like:
    #     if self.log_level < log_level:
    #         return

    if log_stream == LOG_STREAM.FILE_LOG and log_file is None:
        raise ValueError("log_file must be provided if `log_stream` is FILE_LOG")

    from datetime import datetime

    logging_time = datetime.now().time().strftime("%H:%M:%S.%f")[:-3]
    logging_msgs = (
        f"[{logging_time} - {LOG_LEVEL.LOG_TITLE[log_level]}]: {msg}" for msg in msgs
    )

    if log_stream == LOG_STREAM.STRING_LOG:
        return f"{sep.join(logging_msgs)}{end}"

    print(*logging_msgs, sep=sep, end=end, file=log_file)


def log_error(
    *error_msgs, sep="\n", end="\n", log_stream=LOG_STREAM.STDOUT_LOG, log_file=None
):
    return log_msgs(
        *error_msgs,
        sep=sep,
        end=end,
        log_level=LOG_LEVEL.ERROR_LOG,
        log_stream=log_stream,
        log_file=log_file,
    )


def log_str_error(*error_msgs, sep="\n", end=""):
    return log_error(
        *error_msgs,
        sep=sep,
        end=end,
        log_stream=LOG_STREAM.STRING_LOG,
        log_file=None,
    )


def log_info(
    *info_msgs, sep="\n", end="\n", log_stream=LOG_STREAM.STDOUT_LOG, log_file=None
):
    return log_msgs(
        *info_msgs,
        sep=sep,
        end=end,
        log_level=LOG_LEVEL.INFO_LOG,
        log_stream=log_stream,
        log_file=log_file,
    )


def log_str_info(*info_msgs, sep="\n", end=""):
    return log_info(
        *info_msgs,
        sep=sep,
        end=end,
        log_stream=LOG_STREAM.STRING_LOG,
        log_file=None,
    )


def log_debug(
    *debug_msgs, sep="\n", end="\n", log_stream=LOG_STREAM.STDOUT_LOG, log_file=None
):
    return log_msgs(
        *debug_msgs,
        sep=sep,
        end=end,
        log_level=LOG_LEVEL.DEBUG_LOG,
        log_stream=log_stream,
        log_file=log_file,
    )


def log_str_debug(*debug_msgs, sep="\n", end=""):
    return log_debug(
        *debug_msgs,
        sep=sep,
        end=end,
        log_stream=LOG_STREAM.STRING_LOG,
        log_file=None,
    )
