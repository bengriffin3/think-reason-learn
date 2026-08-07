"""Calibrate per-method wall-clock timings on this machine.

Runs each of the four reasoning methods (PI, RRF, GPTree, RRM) on a fresh
cache with a stratified n=100-row slice of the target dataset, measures
wall-clock per method, and extrapolates linearly to the full n to estimate
full-run cost on the current hardware. Results are written to
``precomputed/timings.json`` as::

    {
      "machine": "<hostname / chip>",
      "model": "qwen2.5-coder:14b",
      "dataset": "<vcbench|movie>",
      "n_calibration": 100,
      "methods": {
        "<method>": {
          "calibration_wall_s": ...,
          "calls_made": ...,
          "s_per_call": ...,
          "extrapolated_full_run_s": ...
        },
        ...
      },
      "captured": "<ISO timestamp>"
    }

This replaces guesswork: historical file-timestamp spans include machine
sleep/idle and multi-run appends, so they over- or under-state true
throughput. A short measured run is the only trustworthy basis for the
README's runtime table.

NOTE: implementation lands in Stage 6. Do not run before then.
"""
