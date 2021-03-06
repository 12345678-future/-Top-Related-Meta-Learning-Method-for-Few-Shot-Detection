# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import sys
import tensorflow as tf
import numpy as np
import scipy.misc 
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x


class Logger(object):
    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)
        
    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
def set_logger(path):
    if path:
        #from logger import Logger
        #date = time.strftime("%m_%d_%H_%M") + '_log'
        #log_path = './logs/'+ date
        #if os.path.exists(path):
        #    shutil.rmtree(path)
        #os.makedirs(log_path)
        logger = Logger(path)
        return logger
    else:
        pass
def get_readlines(txt):
    with open(txt,'r') as f:
        content=f.readlines()
    f.close()
    return content

import tensorflow as tf
from numpy import random

writer_1 = tf.summary.FileWriter("./plot/tensorboard_train/author")
writer_2 = tf.summary.FileWriter("./plot/tensorboard_train/rank")
 
log_var = tf.Variable(0.0)
tf.summary.scalar("loss", log_var)
 
write_op = tf.summary.merge_all()
 
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
author_content=get_readlines(sys.argv[1])
rank_content=get_readlines(sys.argv[2])
iteration=0
for c in author_content:
    # for writer 1
    #print(c)
    summary = session.run(write_op, {log_var:float(c.strip('\n').split(" ")[-2])})
    writer_1.add_summary(summary, iteration)
    writer_1.flush()
 
    # for writer 2
    summary = session.run(write_op, {log_var:float(rank_content[iteration].strip('\n').split(" ")[-2])})
    writer_2.add_summary(summary, iteration)
    writer_2.flush()
    iteration+=1
"""
if __name__=="__main__":
    save=sys.argv[1]
    author=sys.argv[2]#author
    rank=sys.argv[3]#rank
    log=set_logger("./plot/tenorboard_train/"+save)
    
    author_content=get_readlines(author) 
    rank_content=get_readlines(rank)
    iteration=0
    for c in content:
        #info_dict={}
        #for tag,value in info_dict.items():
        value=str(c.strip('\n').split(" ")[-1])
        log.scalar_summary("loss", value, iteration)
        
        iteration+=1
"""
