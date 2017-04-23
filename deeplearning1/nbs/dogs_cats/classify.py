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

batches = vgg.get_batches("data/test", batch_size = batch_size)

total = 12501
batch_num = 0
rang = map(lambda x: x + 1, range(batch_size))
count = 0

with open("results.csv", 'wb') as f:
    f.write(b'id,label\n')
    for (imgs,labels) in batches:
        if count >= total:
            break
        else:
            results = vgg.predict(imgs)
            offset = batch_size * batch_num
            ids = map(lambda x: x + offset, rang)
            results = zip(ids, results[0])
            np.savetxt(f, results, delimiter=",", fmt='%.2f')
            count += len(imgs)
        batch_num += 1
        print("On Count " + str(count))
    f.close()

print "done"
