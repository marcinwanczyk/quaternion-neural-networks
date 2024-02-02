from PIL import Image
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import cm
import imageio.v3 as iio

plt.ioff()
category = {0: 'background',
            1: 'avalanche',
            2: 'building_undamaged',
            3: 'building_damaged',
            4: 'cracks/fissure/subsidence',
            5: 'debris/mud/rock flow',
            6: 'fire/flare',
            7: 'flood/water/river/sea',
            8: 'ice_jam_flow',
            9: 'lava_flow',
            10: 'person',
            11: 'pyroclastic_flow',
            12: 'road/railway/bridge',
            13: 'vehicle'}

def viz_segmentation(image):

    fig, ax = plt.subplots()
    n = 14
    # https://matplotlib.org/stable/gallery/color/named_colors.html
    catcolor = {0:'gainsboro',
                1:'paleturquoise',
                2:'rosybrown',
                3:'lightcoral',
                4:'black',
                5:'saddlebrown',
                6:'yellow',
                7:'aqua',
                8:'dodgerblue',
                9:'red',
               10:'lime',
               11:'crimson',
               12:'slategray',
               13:'darkviolet'}

    # cmap = cm.tab20
    colors = [catcolor[x] for x in range(n)]
    cmap = mpl.colors.ListedColormap(colors)
    bounds = [x for x in range(n+1)]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cax = ax.imshow(image, cmap=cmap)
    cax.set_clim(-0.5, 13.5)
    cbar = fig.colorbar(
        cax,
        ticks=[x for x in range(n)], spacing='proportional'
    )
    cbar.ax.set_yticklabels([ '%02d: '%x + category[x] for x in range(n)])
    return fig


def save_viz_segmentation(out_img, out_img_path):
    fig = viz_segmentation(out_img)
    fig.savefig(out_img_path.replace('img_out', 'viz_out'))
    plt.close(fig)                        
    out_img = Image.fromarray(out_img, mode='L')
    out_img.save(out_img_path)



        
