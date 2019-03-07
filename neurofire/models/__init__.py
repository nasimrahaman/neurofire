from .cantor.cantor import Cantor

from .unet.unet_2d import UNet2D
from .unet.unet_2d_5layers import UNet2D5l
from .unet.unet_2d_5l_dilated import UNetDilated2D5l
from .unet.unet_2d_dilated_global import UNetDilated2DGlobal
from .unet.unet_2p5d import UNet2p5D
from .unet.unet_3d import UNet3D
from .unet.unet_3d_2l import UNet3D2l

from .unet_multiscale.unet_2d_multiscale import UNet2DMultiscale
from .unet_multiscale.unet_3d_multiscale import UNet3DMultiscale

from .fcn.fcn import FCN

from .hed.hed import HED
from .hed.fusionhed import FusionHED
from .hed.m2fcn import M2FCN

from .wnet.wnet_2d import WNet2D


def get_model(name):
    assert name in globals(), "Model not found."
    return globals().get(name)
