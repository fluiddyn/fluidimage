
To compile::

  jupyter-nbconvert fluidimage_legi_20170403.ipynb --to slides

  jupyter-nbconvert fluidimage_legi_20170403.ipynb --to slides --post serve

Replace <meta charset="utf-8" /> by <meta http-equiv="Content-Type"
content="text/html; charset=utf-8" />::

  sed -i 's$<meta charset="utf-8" />$<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />$g' sondage_LEGI_2017.slides.html
