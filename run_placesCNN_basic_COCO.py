import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
import glob
import tqdm
import sys

#############################################################
input_dir = "/home/hlin/projects/datasets/mscoco/train2017"
input_files = glob.glob(os.path.join(input_dir, "*"))
output_file = 'mscoco-train2017-resnet50-places365.txt'

# Write out every write_interval iterations
# For debug purposes.
write_interval = 100
#############################################################

# Check if GPUs are available
assert torch.cuda.is_available()
# Only show GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# the architecture to use
arch = 'resnet50'

# load the pre-trained weights
model_file = '%s_places365.pth.tar' % arch
if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)

model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()
model.cuda()  # use gpu

# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load the class label
file_name = 'categories_places365.txt'
if not os.access(file_name, os.W_OK):
    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    os.system('wget ' + synset_url)
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)


outfp = open(output_file, 'w+')
write_count = 0
for img_name in tqdm.tqdm(input_files):
    write_count += 1
    grayscale = False

    img = Image.open(img_name)
    if img.mode == 'L':
        # Image is grayscale.
        grayscale = True
        img = Image.open(img_name).convert('RGB')

    # .type(...) is for gpu
    input_img = V(centre_crop(img).unsqueeze(0)).type(torch.cuda.FloatTensor)

    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    # output the prediction

    # output format is: img_name;grayscale?;prob1,class1;prob2,class2;prob3,class3;prob4,class4;prob5,class5;
    # Write to file.
    line = ''
    img_basename = os.path.basename(img_name)
    line += '{}'.format(img_basename)
    line += ';{}'.format(grayscale)
    for i in range(0, 5):
        line += ';{:.3f},{}'.format(probs[i], classes[idx[i]])
    line += '\n'
    outfp.write(line)

    if write_count % write_interval == 0:
        print('{} prediction on {}'.format(arch,img_name))
        print('grayscale? (1 channel image): {}').format(grayscale)
        for i in range(0, 5):
            print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
        print 'Line written to file: {}'.format(line)

outfp.close()
