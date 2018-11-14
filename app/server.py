from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO
import sys
import codecs
from PIL import Image
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
matching = [s for s in sys.path if "app" in s]
if len(matching)>0:
    appendThis = (matching[0][:-4])
sys.path.append(appendThis)

import tensorflow as tf
tf.enable_eager_execution()

# from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import numpy as np
from DataDownloader.mscocodownloader import MSCocoDownloader
from Models.inceptionmodel import InceptionModel
from Utils.visualiser import Visualiser
from DatasetGeneration.datasetgeneratornolabels import DatasetGeneratorNoLabels
from DatasetGeneration.datasetgenerator import DatasetGenerator
from Models.attentionmodel import AttentionModel
from Models.Helpers.tokenisations import Tokenisations
from pathlib import Path

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


BATCH_SIZE = 128
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512

attention_features_shape = 64

inmodel = InceptionModel()
tokenisations = Tokenisations(True, "configs/tokenizer.pkl", "configs/token_dict.npy")
vis = Visualiser()
attentionModel = AttentionModel(tokenisations, embedding_dim, units, len(tokenisations.tokenizer.word_index), BATCH_SIZE, attention_features_shape, inmodel.image_features_extract_model)

async def download_file(url, dest): 
    if not os.path.exists(sys.path[-1]+'/training_checkpoints3/'):
        os.mkdir(sys.path[-1]+'/training_checkpoints3/')
    if dest.exists():
    	# print("Downloaded file already") 
    	return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_download():
    await download_file("https://www.dropbox.com/s/evkwvtko4u5l9n9/checkpoint?dl=1", Path(sys.path[-1]+'/training_checkpoints2/'+'checkpoint'))  
    await download_file("https://www.dropbox.com/s/d6ec3mkatwqqu11/ckpt.index?dl=1", Path(sys.path[-1]+'/training_checkpoints2/'+'ckpt.index'))
    await download_file("https://www.dropbox.com/s/nsu98bb622kkpai/ckpt.data-00000-of-00001?dl=1", Path(sys.path[-1]+'/training_checkpoints2/'+'ckpt.data-00000-of-00001'))

    return None

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_download())]
loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

# vis.plot_attention(image_path, result, attention_plot)
# opening the image
# Image.open(image_path)

# def setup_learner():
#     inmodel = InceptionModel()
#     tokenisations = Tokenisations(True, "Configs/tokenizer.pkl", "Configs/token_dict.npy")
#     vis = Visualiser()
#     attentionModel = AttentionModel(tokenisations, embedding_dim, units, len(tokenisations.tokenizer.word_index), BATCH_SIZE, attention_features_shape, inmodel.image_features_extract_model)
#     print(sys.path)
#     attentionModel.load_checkpoint("training_checkpoints")
#     return attentionModel

# # loop = asyncio.get_event_loop()
# # tasks = [asyncio.ensure_future(setup_learner())]
# # attentionModel = loop.run_until_complete(asyncio.gather(*tasks))[0]
# # loop.close()
# attentionModel = setup_learner()

@app.route('/')
def index(request):
    html=codecs.open("app/view/index.html", 'r')
    return HTMLResponse(html.read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    image_path = np.array(Image.open(BytesIO(img_bytes)).resize((299,299)))[:,:,0:3]
#     image_url = 'https://tensorflow.org/images/surf.jpg'
#     image_extension = image_url[-4:]
#     image_path = tf.keras.utils.get_file('image'+image_extension, 
#                                          origin=image_url)
    
    result, attention_plot = attentionModel.evaluate2(image_path)

    return JSONResponse({'result':' '.join(result)})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=5042)

