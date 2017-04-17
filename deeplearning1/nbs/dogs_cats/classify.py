"""Thing"""
import vgg16; reload(vgg16)
import numpy as np
from vgg16 import Vgg16

vgg = Vgg16()
path = "data/sample/"
batch_size = 64

batches = vgg.get_batches(path + "train", batch_size = batch_size)
val_batches = vgg.get_batches(path + "valid", batch_size = batch_size)

vgg.finetune(batches)
vgg.fit(batches, val_batches, nb_epoch = 1)

imgs,labels = next(batches)


results = vgg.predict(imgs)
results = zip(results[0], results[1])
results.insert(0, ("id", "label"))
np.savetxt("results.csv", results, delimiter=",")
print "done"
