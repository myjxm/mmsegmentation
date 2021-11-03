from mmseg.apis import inference_segmentor, init_segmentor,inference_segmentor_concactdata
import mmcv
from skimage import io,data,color,img_as_ubyte
import numpy as np

def gamma(gamma,file_path):
    gamma = gamma
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype('uint8')
    # test a single image and show the results
    img = file_path  # or img = mmcv.imread(img), which will only load it once
    image = io.imread(img)
    results = mmcv.lut_transform(image, table)
    return results



img = '/home/zrd/segment-val/sythony-lake-review-sample-160201-0001-0742.jpg'  # or img = mmcv.imread(img), which will only load it once

results = gamma(1.0,img)
io.imsave('/home/home2/zrd/gamma/sythony-lake-review-sample-160201-0001-0742-gamma1.0.jpg', results)


results = gamma(0.5,img)
io.imsave('/home/home2/zrd/gamma/sythony-lake-review-sample-160201-0001-0742-gamma0.5.jpg', results)

results = gamma(2,img)
io.imsave('/home/home2/zrd/gamma/sythony-lake-review-sample-160201-0001-0742-gamma2.jpg', results)






