# Code as for a real PIV project

Here, we gather modules and scripts for preprocessing and PIV computation as we
would do for a real project. This can be used as a first step when you start a
new project of PIV computation with fluidimage.

```{eval-rst}
+---------------+------------------------------------------------------+
| args.py       | code to define command line tools with argparse      |
+---------------+------------------------------------------------------+
| params_pre.py | parameters for the image preprocessing               |
+---------------+------------------------------------------------------+
| job_pre.py    | job for image preprocessing                          |
+---------------+------------------------------------------------------+
| submit_pre.py | submit preprocessing jobs on a cluster               |
+---------------+------------------------------------------------------+
| params_piv.py | parameters for the PIV computation                   |
+---------------+------------------------------------------------------+
| try_piv.py    | script to be used with ipython to try piv parameters |
+---------------+------------------------------------------------------+
| job_piv.py    | job for PIV computation                              |
+---------------+------------------------------------------------------+
| submit_piv.py | submit PIV jobs on a cluster                         |
+---------------+------------------------------------------------------+
```

## args.py

```{literalinclude} args.py
```

## params_pre.py

```{literalinclude} params_pre.py
```

## job_pre.py

```{literalinclude} job_pre.py
```

## submit_pre.py

```{literalinclude} submit_pre.py
```

## params_piv.py

```{literalinclude} params_piv.py
```

## try_piv.py

```{literalinclude} try_piv.py
```

## job_piv.py

```{literalinclude} job_piv.py
```

## submit_piv.py

```{literalinclude} submit_piv.py
```
