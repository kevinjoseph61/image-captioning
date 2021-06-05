import tensorflow as tf
from queue import Queue
from .inference import Inference
from .model_train import ModelConfig, Vocabulary, CaptionGenerator
from .models import Request, Image, Caption
from image_cap.settings import BASE_DIR
from os import path
from math import exp

class Model():
    def __init__(self):
        self.request_queue = Queue()
        self.g = None
        self.restore_fn = None
        self.vocab = None
        self.model = None
        self.checkpoint_path = path.join(BASE_DIR, "model", "model.ckpt-first")
        self.vocab_file = path.join(BASE_DIR, "model", "words_cnt.txt")
    
    def build(self):
        #tf.logging.set_verbosity(tf.logging.FATAL)
        self.g = tf.Graph()
        with self.g.as_default():
            self.model = Inference()
            self.restore_fn = self.model.build_graph_from_config(ModelConfig(), self.checkpoint_path)
        self.g.finalize()
        self.vocab = Vocabulary(self.vocab_file)

        # filenames = []
        # for file_pattern in input_files.split(","):
        #     filenames.extend(tf.gfile.Glob(file_pattern))
        # tf.logging.info("Running caption generation on %d files matching %s",
        #                 len(filenames), input_files)

def model_thread_target(requestID, input_image_list):
    tf.logging.set_verbosity(tf.logging.FATAL)
    req = Request.objects.get(pk=requestID)
    req.status = 2
    req.save()
    checkpoint_path = path.join(BASE_DIR, "interface", "model", "model.ckpt-first")
    vocab_file = path.join(BASE_DIR, "interface", "model", "words_cnt.txt")
    g = tf.Graph()
    with g.as_default():
        model = Inference()
        restore_fn = model.build_graph_from_config(ModelConfig(), checkpoint_path)
    g.finalize()
    vocab = Vocabulary(vocab_file)
    images = []
    for i, file_pattern in input_image_list:
        images.extend(tf.gfile.Glob(file_pattern))
    with tf.Session(graph=g) as sess:
        restore_fn(sess)
        generator = CaptionGenerator(model, vocab)
        captionlist = []
        for i, filename in enumerate(images):
            with tf.gfile.GFile(filename, "rb") as f:
                image = f.read()
            im = Image.objects.get(pk=input_image_list[i][0])
            captions = generator.beam_search(sess, image)
            for i, caption in enumerate(captions):
                sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
                sentence = " ".join(sentence)
                print("  %d) %s (p=%f)" % (i, sentence, exp(caption.logprob)))
                captionlist.append(sentence)
                cap = Caption(image=im, caption=sentence, probability=exp(caption.logprob))
                cap.save()
    req.status = 3
    req.save()
