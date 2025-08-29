import matplotlib.pyplot as plt
import sys
sys.path.append('/home/ana/pyISSM/src')
import pyissm as issm

ana = issm.model.io.load_model('flowline9_profile_002.nc')

ana_mesh, ana_x, ana_y, ana_elements, ana_is3d = issm.model.mesh.process_mesh(ana)


# issm.plot.plot_mesh2d(ana_mesh, show_nodes = True)
issm.plot.plot_model_nodes(ana, ana.mask.ice_levelset, ana.mask.ocean_levelset, s=10, type='ice_front_nodes')

plt.show()


