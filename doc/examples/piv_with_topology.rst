PIV computation 
===============

This minimal example presents how to carry out a simple PIV computation.  See
also the documentation of the class
:class:`fluidimage.topologies.piv.TopologyPIV` and the work defined in the
subpackage :mod:`fluidimage.works.piv`.

.. literalinclude:: piv_with_topology.py

We now show a similar example but with a simple preprocessing (using a function
``im2im``):

.. literalinclude:: piv_with_topo_and_preproc.py

The file ``my_example_im2im.py`` should be importable (for example in the same
directory than ``piv_with_topo_and_preproc.py``)

.. literalinclude:: my_example_im2im.py

Same thing but the preprocessing is done with a class ``Im2Im``

.. literalinclude:: piv_with_topo_and_preproc_class.py

The file ``my_example_im2im_class.py`` should be importable (for example in the
same directory than ``piv_with_topo_and_preproc_class.py``)

.. literalinclude:: my_example_im2im_class.py
