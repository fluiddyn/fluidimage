# PIV computation on a cluster

This minimal example presents how to carry out a simple PIV computation on a
cluster. We need a script for the computation and a script to submit the job.

We call the script for the computation `piv_complete.py`. For idempotent job,
it is important to set `params.saving.how = 'complete'`.

```{literalinclude} piv_parallel_complete.py
```

The submission script is quite simple:

```{literalinclude} submit_job_legi.py
```
