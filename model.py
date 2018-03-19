from utils import *
from line_drawing_utils import *

class Shader():
    def __init__(self, input_size=[256, 256, 1][:], output_size=[256, 256, 1][:], batch_size=4):
        self.batch_size = batch_size
        self.output_size = output_size

        self.l1_loss_wt = 100

        self.gen_dim_mult = 64
        self.disc_dim_mult = 64

        self.line_images = tf.placeholder(tf.float32, [self.batch_size] + input_size)
        self.real_images = tf.placeholder(tf.float32, [self.batch_size] + output_size)

        self.gen_shaded_images = self.generator(self.line_images)

        self.real_images_full = tf.concat([self.line_images, self.real_images], 3)
        self.fake_images_full = tf.concat([self.line_images, self.gen_shaded_images], 3)

        # We reuse the discriminator when its run the second time because we need the
        # same Variables (and thus the same network) used both times
        self.disc_real_logits = self.discriminator(self.real_images_full, reuse=False)
        self.disc_fake_logits = self.discriminator(self.fake_images_full, reuse=True)

        # self.d_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.disc_real_logits, labels=tf.stop_gradient(tf.constant([1]*batch_size))))
        # self.d_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.disc_fake_logits, labels=tf.constant([0]*batch_size)))
        # self.d_loss = self.d_loss_real + self.d_loss_fake
        #
        # self.g_adv_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.disc_fake_logits, labels=tf.stop_gradient(tf.constant([1]*batch_size))))
        # self.g_l1_loss = tf.reduce_mean(tf.abs(self.real_images - self.gen_shaded_images))
        # self.g_loss = self.g_adv_loss + self.l1_loss_wt * self.g_l1_loss

        self.d_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.disc_real_logits, labels=tf.ones_like(self.disc_real_logits)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.disc_fake_logits, labels=tf.zeros_like(self.disc_fake_logits)))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_adv_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.disc_fake_logits, labels=tf.ones_like(self.disc_fake_logits)))
        # self.g_l1_loss = tf.reduce_mean(tf.abs(self.real_images - self.gen_shaded_images))
        self.g_l1_loss = tf.reduce_mean(tf.nn.l2_loss(self.real_images - self.gen_shaded_images))
        self.g_loss = self.g_adv_loss + self.l1_loss_wt * self.g_l1_loss


        # https://stackoverflow.com/questions/44578992/how-to-update-the-variable-list-for-which-the-optimizer-need-to-update-in-tensor
        tf_vars = tf.trainable_variables()
        self.d_vars = [var for var in tf_vars if 'd_' in var.name]
        self.g_vars = [var for var in tf_vars if 'g_' in var.name]

        # The variables are separately addressed by the optimizers. When sess.run() is run on a loss
        # function, only the corresponding variables get backpropogated into
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            self.d_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
            self.g_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)

    def discriminator(self, image, y=None, reuse=False):
        # image is 256 x 256 x (input_c_dim + output_c_dim)
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        h0 = lrelu(conv2d(image, self.disc_dim_mult, name='d_h0_conv')) # h0 is (128 x 128 x self.disc_dim_mult)
        h1 = lrelu(bn(conv2d(h0, self.disc_dim_mult*2, name='d_h1_conv'))) # h1 is (64 x 64 x self.disc_dim_mult*2)
        h2 = lrelu(bn(conv2d(h1, self.disc_dim_mult*4, name='d_h2_conv'))) # h2 is (32 x 32 x self.disc_dim_mult*4)
        h3 = lrelu(bn(conv2d(h2, self.disc_dim_mult*8, stride_h=1, stride_w=1, name='d_h3_conv'))) # h3 is (16 x 16 x self.disc_dim_mult*8)
        h4 = dense(tf.reshape(h3, [self.batch_size, -1]), 1, activation=None)
        return h4

    def generator(self, img_in):
        s = self.output_size[0]
        s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)
        # image is (256 x 256 x input_c_dim)
        e1 = conv2d(img_in, self.gen_dim_mult, name='g_e1_conv') # e1 is (128 x 128 x self.gen_dim_mult)
        e2 = bn(conv2d(lrelu(e1), self.gen_dim_mult*2, name='g_e2_conv')) # e2 is (64 x 64 x self.gen_dim_mult*2)
        e3 = bn(conv2d(lrelu(e2), self.gen_dim_mult*4, name='g_e3_conv')) # e3 is (32 x 32 x self.gen_dim_mult*4)
        e4 = bn(conv2d(lrelu(e3), self.gen_dim_mult*8, name='g_e4_conv')) # e4 is (16 x 16 x self.gen_dim_mult*8)
        e5 = bn(conv2d(lrelu(e4), self.gen_dim_mult*8, name='g_e5_conv')) # e5 is (8 x 8 x self.gen_dim_mult*8)

        self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(e5), [self.batch_size, s16, s16, self.gen_dim_mult*8], name='g_d4', with_w=True)
        d4 = bn(self.d4)
        d4 = tf.concat([d4, e4], 3)
        # d4 is (16 x 16 x self.gen_dim_mult*8*2)

        self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4), [self.batch_size, s8, s8, self.gen_dim_mult*4], name='g_d5', with_w=True)
        d5 = bn(self.d5)
        d5 = tf.concat([d5, e3], 3)

        # d5 is (32 x 32 x self.gen_dim_mult*4*2)
        self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5), [self.batch_size, s4, s4, self.gen_dim_mult*2], name='g_d6', with_w=True)
        d6 = bn(self.d6)
        d6 = tf.concat([d6, e2], 3)
        # d6 is (64 x 64 x self.gen_dim_mult*2*2)

        self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6), [self.batch_size, s2, s2, self.gen_dim_mult], name='g_d7', with_w=True)
        d7 = bn(self.d7)
        d7 = tf.concat([d7, e1], 3)
        # d7 is (128 x 128 x self.gen_dim_mult*1*2)

        self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7), [self.batch_size, s, s, self.output_size[-1]], name='g_d8', with_w=True)
        # d8 is (256 x 256 x output_c_dim)

        return tf.nn.tanh(self.d8)

    def loadmodel(self, load_discrim=True):
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

        if load_discrim:
            self.saver = tf.train.Saver()
        else:
            self.saver = tf.train.Saver(self.g_vars)

        if self.load("./checkpoint"):
            print("Loaded")
        else:
            print("Load failed")

    def save(self, checkpoint_dir, step):
        model_name = "model"
        model_dir = "tr"

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "tr"
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def train(self):
        self.loadmodel()

        data = get_image_dirs()
        print(data[0])
        base = np.array([get_image(sample_file) for sample_file in data[0: self.batch_size]])
        base_grayscale = np.array([get_grayscale(ba) for ba in base]) / 255.0

        base_edge = np.array([get_line_drawing(img) for img in base]) / 255.0
        base_edge = np.expand_dims(base_edge, 3)

        base_normalized = base/255.0

        for i in xrange(self.batch_size):
            ## all are normalized
            imwrite("results/base_" + str(i) + ".png", base_normalized[i])
            imwrite("results/base_line_" + str(i) + ".jpg", base_edge[i])
            imwrite("results/base_grayscale_" + str(i) + ".png", base_grayscale[i])

        datalen = len(data)

        d_loss_tot = 0.0

        for e in xrange(20000):
            avg_d_loss = d_loss_tot / (datalen / self.batch_size)
            d_loss_tot = 0.0
            if avg_d_loss > 0.1:
                print("Training discriminator")
            for i in xrange(datalen / self.batch_size):
                batch_files = data[i * self.batch_size : (i+1) * self.batch_size]
                batch = np.array([get_image(batch_file) for batch_file in batch_files])

                batch_grayscale = np.array([get_grayscale(ba) for ba in batch]) / 255.0
                batch_grayscale = np.expand_dims(batch_grayscale, 3)

                batch_edge = np.array([get_line_drawing(img) for img in batch]) / 255.0
                batch_edge = np.expand_dims(batch_edge, 3)

                batch_normalized = batch/255.0

                if avg_d_loss > 0.1:
                    d_loss, _ = self.sess.run([self.d_loss, self.d_optim], feed_dict={self.line_images: batch_edge,
                                                                                      self.real_images: batch_grayscale})
                else:
                    d_loss = self.sess.run(self.d_loss, feed_dict={self.line_images: batch_edge,
                                                                                      self.real_images: batch_grayscale})
                d_loss_tot += d_loss
                g_loss, _ = self.sess.run([self.g_loss, self.g_optim], feed_dict={self.line_images: batch_edge, self.real_images: batch_grayscale})

                print("%d: [%d / %d] d_loss %f, g_loss %f, avg_d_loss %f" % (e, i, (datalen/self.batch_size), d_loss, g_loss, avg_d_loss))

                if i % 100 == 0:
                    recreation = self.sess.run(self.gen_shaded_images, feed_dict={self.line_images: batch_edge, self.real_images: batch_grayscale})
                    for j in xrange(self.batch_size):
                        imwrite("results/"+str(e*100000 + i) + "_" + str(j) +".jpg", recreation[j])


                if i % 500 == 0:
                    self.save("./checkpoint", e*100000 + i)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python model.py [train, sample]")
    else:
        cmd = sys.argv[1]
        if cmd == "train":
            c = Shader()
            c.train()
        else:
            print("Usage: python model.py train")