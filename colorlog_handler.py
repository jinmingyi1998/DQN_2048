# -*- coding: utf-8 -*-
# @File : color.py
# @Author : r.yang
# @Date : Sun Nov 28 20:35:16 2021
# @Description : color log handler


def ColorLoggerHandler():
    import colorlog

    fmt = "%(asctime)s %(name)s(%(process)d) %(filename)s:%(lineno)d %(funcName)s [%(levelname)s] %(message)s"

    handler = colorlog.StreamHandler()
    log_color = {
        "DEBUG": "bold_cyan",
        "INFO": "bold_green",
        "WARNING": "bold_yellow",
        "ERROR": "bold_red",
        "CRITICAL": "bg_white,bold_red",
    }
    handler.setFormatter(
        colorlog.ColoredFormatter(
            fmt.replace("%(asctime)s", "%(green)s%(asctime)s%(reset)s")
            .replace("%(name)s", "%(blue)s%(name)s%(reset)s")
            .replace("%(levelname)s", "%(log_color)s%(levelname)s%(reset)s")
            .replace(
                "%(filename)s:%(lineno)d", "%(cyan)s%(filename)s:%(lineno)d%(reset)s"
            ),
            # datefmt='%Y-%m-%d %H:%M:%S',
            log_colors=log_color,
        )
    )
    return handler
