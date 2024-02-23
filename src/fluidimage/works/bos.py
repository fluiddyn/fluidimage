"""Background-oriented schlieren

Provides

.. autoclass:: WorkBOS
   :members:
   :private-members:

"""

from pathlib import Path

from fluidimage.data_objects.piv import ArrayCoupleBOS
from fluidimage.util import imread
from fluidimage.works import BaseWorkFromImage
from fluidimage.works.piv import WorkPIV


class WorkBOS(BaseWorkFromImage):
    """Work for BOS computation.

    See https://en.wikipedia.org/wiki/Background-oriented_schlieren_technique

    """

    path_dir_src: Path

    @classmethod
    def _complete_params_with_default(cls, params):
        super()._complete_params_with_default(params)
        WorkPIV._complete_params_with_default_piv(params)

        params._set_attrib("reference", 0)

        params._set_doc(
            """
reference : str or int, {0}

    Reference file (from which the displacements will be computed). Can be an
    absolute file path, a file name or the index in the list of files found
    from the parameters in ``params.images``.

"""
        )

    def __init__(self, params):
        super().__init__(params)
        self.init_from_input()
        self.work_piv = WorkPIV(params)

    def init_from_input(self):
        self._init_serie()
        path_reference = self._init_path_reference(self.params)
        self.name_reference = path_reference.name
        self.path_reference = path_reference
        self.image_reference = imread(path_reference)

    def _init_path_reference(self, params):
        reference = params.reference
        self.path_dir_src = Path(self.serie.path_dir).absolute()

        if isinstance(reference, int):
            names = self.serie.get_name_arrays()
            names = sorted(names)
            path_reference = self.path_dir_src / names[reference]
        else:
            reference = Path(reference).expanduser()
            if reference.is_file():
                path_reference = reference
            else:
                path_reference = self.path_dir_src / reference
                if not path_reference.is_file():
                    raise ValueError(
                        "Bad value of params.reference:" + path_reference
                    )
        return path_reference

    def calcul(self, tuple_image_name):
        """Calcul BOS from one image"""

        image, name = tuple_image_name
        path = self.path_dir_src / name

        array_couple = ArrayCoupleBOS(
            names=(self.name_reference, name),
            arrays=(self.image_reference, image),
            params_mask=self.params.mask,
            serie=self.serie,
            paths=[self.path_reference, path],
        )
        return self.work_piv.calcul(array_couple)
