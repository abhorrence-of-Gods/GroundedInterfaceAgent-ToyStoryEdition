from __future__ import annotations

"""Human-Feedback interface for Like/Dislike buttons.

This module spins up a lightweight Flask HTTP server that exposes two endpoints::

    POST /like
    POST /dislike

Each endpoint enqueues a (timestamp, value) pair into a multiprocessing ``Queue``
that can be consumed by the training process. ``value`` is ``1`` for *Like*, ``0``
for *Dislike*.

Usage
-----
>>> from environments.human_feedback_interface import start_feedback_server
>>> queue = start_feedback_server(port=8001)
>>> # In training loop:
>>> while not queue.empty():
>>>     ts, val = queue.get()
>>>     ...  # feed val into replay buffer as r_human
"""

from multiprocessing import Process, Queue
from datetime import datetime
from typing import Tuple

from flask import Flask, request, jsonify

__all__ = ["start_feedback_server"]


def _create_app(queue: Queue) -> Flask:
    app = Flask(__name__)

    @app.route("/like", methods=["POST"])
    def like():
        queue.put((datetime.utcnow().timestamp(), 1))
        return jsonify(status="ok", value=1)

    @app.route("/dislike", methods=["POST"])
    def dislike():
        queue.put((datetime.utcnow().timestamp(), 0))
        return jsonify(status="ok", value=0)

    return app


def _run_server(queue: Queue, host: str, port: int):
    app = _create_app(queue)
    # Flask built-in server is fine for prototype (not for prod)
    app.run(host=host, port=port, debug=False, use_reloader=False)


def start_feedback_server(host: str = "0.0.0.0", port: int = 8001) -> Queue:
    """Launch the feedback HTTP server in a background process.

    Returns
    -------
    multiprocessing.Queue
        Queue where (timestamp, value) tuples are pushed.
    """
    q: Queue = Queue()
    proc = Process(target=_run_server, args=(q, host, port), daemon=True)
    proc.start()
    return q 