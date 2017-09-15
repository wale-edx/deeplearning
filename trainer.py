import numpy as np
import tensorflow as tf
from tqdm import tqdm

class Trainer:
    def __init__(self,sess, config,model,data,FLAGS):
        self.n_epochs = config['epoch']
        self.batch_size = config['batch_size']
        self.ckpt_path = config['ckpt_path']
        self.save_freq = config['save_freq']
        self.sess = sess
        self.model=model
        Trainer.state = False

    def train(self):
        # init the epoch as a tensor to be saved in the graph so i can restore it and continue traning

        # training
        for cur_epoch in range(self.n_epochs + 1):
            loop=tqdm(range(self.config.nit_epoch))
            for it in loop:
                batch_x, batch_y = self.data.train.next_batch(self.batch_size)
                feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
                _,loss,acc=self.sess.run([self.model.train_step,self.model.cross_entropy,self.model.accuracy],
                                     feed_dict=feed_dict)
               
            loop.close()
            print("epoch-" + str(cur_epoch) + "-" + "loss-" + str(loss))

    def test(self):
        feed_dict = {self.model.x: self.data.test.images, self.model.y: self.data.test.labels, self.model.is_training: False}

        print("Test Acc : ", self.sess.run(self.model.accuracy, feed_dict=feed_dict),
              "% \nExpected to get around 94-98% Acc")
