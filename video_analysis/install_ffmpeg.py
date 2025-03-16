import subprocess as sb
import sys

from video_analysis.utils.logging import log_error, log_info


def install_via_pip_alpine():
    log_info("Starting FFmpeg installation")

    base_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
    ]

    try:
        update_python_cmd = [
            *base_cmd,
            "--upgrade",
            "pip",
            "setuptools",
        ]
        sb.check_call(update_python_cmd)
        log_info("Updated `pip` successfully")

    except Exception as e:
        log_error("failed to update python", e)

    try:
        ffmpeg_install_cmd = [
            *base_cmd,
            "ffmpeg-python",
        ]
        sb.check_call(ffmpeg_install_cmd)
        log_info("Installed `ffmpeg-python` successfully")

    except Exception as e:
        log_error("Failed to install `ffmpeg-python` via pip", e)

    try:
        # this command is exclusive to alpine containers, I'll make it more versatile later
        download_ffmpeg_cmd = ["apk", "add", "--no-cache", "ffmpeg"]
        sb.check_call(download_ffmpeg_cmd)
        log_info("Installed FFmpeg successfully.")

    except Exception as e:
        log_error(
            "Failed to install ffmpeg in the container. Please try to install it manually.",
            e,
        )
