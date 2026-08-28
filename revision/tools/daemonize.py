#!/usr/bin/env python3
"""Detach a command into its own session so process-group kills cannot reach it.

macOS ships no setsid(1), and `nohup cmd &` leaves the child in the caller's
process group — which is how the first final-run driver was killed while its own
child survived. Double-fork + os.setsid() puts the driver in a fresh session with
no controlling terminal.

Usage: daemonize.py <logfile> <command> [args...]
"""
import os
import sys

if len(sys.argv) < 3:
    sys.exit("usage: daemonize.py <logfile> <command> [args...]")

logfile, cmd = sys.argv[1], sys.argv[2:]

if os.fork() > 0:
    os._exit(0)          # parent returns to the shell immediately
os.setsid()              # new session, no controlling terminal
if os.fork() > 0:
    os._exit(0)          # ensure we can never re-acquire a terminal

fd = os.open(logfile, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
os.dup2(fd, 1)
os.dup2(fd, 2)
devnull = os.open(os.devnull, os.O_RDONLY)
os.dup2(devnull, 0)
os.execv(cmd[0], cmd)
