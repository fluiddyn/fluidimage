import unittest
from shutil import rmtree
from pathlib import Path

from fluiddyn.io import stdout_redirected


from fluidimage import path_image_samples


class TTeestSurftrack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path_input_files = path_image_samples / "SurfTracking/Images"
        cls.postfix = "test_surftrack"

    @classmethod
    def tearDownClass(cls):
        path = cls.path_input_files
        path_out = Path(str(path) + "." + cls.postfix)
        if path_out.exists():
            rmtree(path_out)

    def tttest_surftrack(self):
        params = TopologySurfaceTracking.create_default_params()

        params.film.path = str(self.path_input_files)
        params.film.ind_start = 1
        params.film.path_ref = str(self.path_input_files)
        params.surface_tracking.xmin = 125
        params.surface_tracking.xmax = 290
        params.series.ind_start = 1

        # params.saving.how has to be equal to 'complete' for idempotent jobs
        # (on clusters)
        params.saving.plot = False
        params.saving.how_many = 100
        params.saving.how = "complete"
        params.saving.postfix = self.postfix
        print(params)
        topology = TopologySurfaceTracking(params, logging_level="info")
        # topology.make_code_graphviz('topo.dot')
        seq = False
        with stdout_redirected():
            topology = TopologySurfaceTracking(params, logging_level="info")
            topology.compute(sequential=seq)
        #            print(topology.path_dir_result)
        #            log = LogTopology(topology.path_dir_result)
        #        topology.compute(sequential=seq)

        # not generating plots if seq mode is false
        if seq == False:
            params.saving.plot = False


#        log.plot_durations()
#        log.plot_nb_workers()
#        log.plot_memory()


# if __name__ == "__main__":
#    unittest.main()
