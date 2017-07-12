from util import *
from layers import *
from VOC2012_devkit import *
class SegNetModel():

    learning_rate = tf.placeholder(tf.float32, shape=[], name='lr')
    pretrained_weight_path = './pretrained_checkpoint'
    weight_path = './checkpoint'

    def __init__(self, config, img_width, img_height, reuse = None):
        self.threshold = config.threshold
        self.training = config.training
        self.reuse = reuse
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = config.batch_size
        self.init_learning_rate = config.init_learning_rate
        self.learning_rate_decay = config.learning_rate_decay
        self.num_epoch = config.num_epoch
        self.num_class = config.num_class

    def input_pipeline(self, fn_img, fn_seg=None):
        reader = tf.WholeFileReader()      

        with tf.variable_scope('image'):
            fn_img_queue = tf.train.string_input_producer(fn_img, shuffle=False)
            _, value = reader.read(fn_img_queue)
            img = tf.image.decode_jpeg(value, channels=3)
            img = tf.image.resize_images(img, [self.img_height, self.img_width], method=tf.image.ResizeMethod.BILINEAR)
            img = tf.cast(img, dtype = tf.float32)
            img = tf.reshape(img, [self.img_height, self.img_width, 3])
            if self.training is False:
                img = tf.train.batch([img], self.batch_size)
                return img

        if not len(fn_seg) == len(fn_img):
            raise ValueError('Number of images and segmentations do not match!')

        with tf.variable_scope('segmentation'):
            fn_seg_queue = tf.train.string_input_producer(fn_seg, shuffle=False)
            _, value = reader.read(fn_seg_queue)
            seg = tf.image.decode_png(value, channels=1)
            seg = tf.image.resize_images(seg, [self.img_height, self.img_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            seg = tf.reshape(seg, [self.img_height, self.img_width, 1])
            ground_truth = seg
            ground_truth = tf.image.rgb_to_grayscale(ground_truth, name=None)
            seg = ImageToLabel(seg, self.img_height, self.img_width)
            seg = tf.reshape(seg, [self.img_height, self.img_width, self.num_class])
            
        if self.training is True:
            with tf.variable_scope('shuffle'):
                seg, img, ground_truth = tf.train.shuffle_batch([seg, img, ground_truth], batch_size=self.batch_size,
                                                    num_threads=4,
                                                    capacity=1000 + 3 * self.batch_size,
                                                    min_after_dequeue=1000)
        return seg, img, ground_truth

    def chk_point_restore(self, sess):
        saver = tf.train.Saver(max_to_keep=2)
        if self.training:
            ckpt = tf.train.get_checkpoint_state(self.pretrained_weight_path)
        else:
            ckpt = tf.train.get_checkpoint_state(self.weight_path)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            if self.training:
                saver.restore(sess, os.path.join(self.pretrained_weight_path, ckpt_name))
            else:
                saver.restore(sess, os.path.join(self.weight_path, ckpt_name))
            print('[*] Success to read {}'.format(ckpt_name))
        else:
            if self.training:
                print('[*] Failed to find a checkpoint. Start training from scratch ...')
            else:
                raise ValueError('[*] Failed to find a checkpoint.')

        self.saver = saver

    def SegNet_training_setup(self, fn_img, fn_seg):

        y, x, gt = self.input_pipeline(fn_img, fn_seg)
        gt = tf.cast(gt, tf.uint8)
        logits, loss = self.build_model(x, y)

        y = tf.to_int64(y, name = 'y')
        pred_train = 12*tf.cast(logits, tf.uint8)
        gt = tf.reshape(gt, [-1, self.img_height, self.img_width, 1])
        pred_train = tf.reshape(pred_train, [-1, self.img_height, self.img_width, 1])
        result_train = tf.concat([gt, pred_train], axis=2)
        result_train = tf.cast(tf.reshape(result_train, [-1, self.img_height, self.img_width*2, 1]), tf.uint8)

        rgb_image = tf.cast(logits, tf.uint8)
        rgb_image = tf.reshape(rgb_image, [-1, self.img_height, self.img_width, 1])

        rgb_image = LabelToRGB (rgb_image)
        x = tf.cast(x, tf.uint8)
        rgb_image = tf.concat([x, rgb_image], axis=2)
        rgb_image = tf.reshape(rgb_image, [-1, self.img_height, self.img_width*2, 3])
        tf.summary.scalar('loss', loss)
        tf.summary.image('result_train', result_train, max_outputs=self.batch_size)
        tf.summary.image('rgb_image', rgb_image, max_outputs=self.batch_size)
        tf.summary.scalar('learning_rate', self.learning_rate)
    
        sum_all = tf.summary.merge_all()
        
        vars_trainable = tf.trainable_variables()

        num_param = 0

        for var in vars_trainable:
            num_param += prod(var.get_shape()).value
            tf.summary.histogram(var.name, var)

        print('\nTotal nummber of parameters = %d' % num_param)

        train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss, var_list=vars_trainable)

        self.loss = loss
        self.sum_all = sum_all
        self.train_step = train_step
        self.pred_train = pred_train
        self.result_train = result_train
        

    def SegNet_training(self, sess, path_pack):

        total_count = 0
        t0 = time.time()

        if os.path.exists('./logs'):
            import  shutil
            shutil.rmtree('./logs')


        writer = tf.summary.FileWriter("./logs", sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for epoch in range(self.num_epoch):

            lr = self.init_learning_rate * self.learning_rate_decay**(epoch//20000)

            for k in range(self.batch_size // self.batch_size):
                
                l_train, _ = sess.run([self.loss, self.train_step], feed_dict={self.learning_rate: lr})

                writer.add_summary(sess.run(self.sum_all, feed_dict={self.learning_rate: lr}), total_count)
                total_count += 1                
                m, s = divmod(time.time() - t0, 60)
                h, m = divmod(m, 60)
                print('Epoch: [%4d/%4d], [%4d/%4d], Time: [%02d:%02d:%02d], loss: %.4f'
                % (epoch+1, self.num_epoch, k+1, self.batch_size // self.batch_size, h, m, s, l_train))

            if epoch % 100 == 0 and epoch != 0:
                print('Saving checkpoint ...')
                self.saver.save(sess, self.weight_path + '/Davis.ckpt')

        coord.request_stop()         
        coord.join(threads)

    def SegNet_testing_setup(self, fn_img):

        x = self.input_pipeline(fn_img)
        x = tf.reshape(x, [self.batch_size, self.img_height, self.img_width, 3])
        logits = self.build_model(x)

        rgb_image = tf.cast(logits, tf.uint8)
        rgb_image = tf.reshape(rgb_image, [-1, self.img_height, self.img_width, 1])
        rgb_image = LabelToRGB (rgb_image)

        x = tf.cast(x, tf.uint8)
        rgb_image = tf.concat([x, rgb_image], axis=2)
        rgb_image = tf.reshape(rgb_image, [-1, self.img_height, self.img_width*2, 3])

        self.val_result = rgb_image
        self.x = x

    def SegNet_testing(self, sess, path_pack, fn_img):

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        total_count = 1
        for k in range(len(fn_img)):
            result= sess.run([self.val_result])

            result = reshape(result, (-1, self.img_height, 2*self.img_width, 3))

            img_name = os.path.basename(fn_img[k])
            img_name = os.path.splitext(img_name)[0]

            imsave(path_pack.result_dir+'/'+img_name+'.jpg', result[0])
            print 'Saved result image '+img_name
            total_count += 1

        coord.request_stop()         
        coord.join(threads)

    def build_model(self, x, y = None):

        with tf.variable_scope('SegNet'):
            
            x = x[..., ::-1] - [103.939, 116.779, 123.68]

            # 224 224
            conv1 = conv_relu_vgg(x, name='conv1_1')
            conv1 = conv_relu_vgg(conv1, name='conv1_2')

            # 112 112
            #pool1 = tf.layers.max_pooling2d(conv1, 2, 2, name='pool1')
            pool1, max1 = max_pooling(conv1, 2, 2, name='pool1')
            conv2 = conv_relu_vgg(pool1, name='conv2_1')
            conv2 = conv_relu_vgg(conv2, name='conv2_2')

            # 56 56
            #pool2 = tf.layers.max_pooling2d(conv2, 2, 2, name='pool2')
            pool2, max2 = max_pooling(conv2, 2, 2, name='pool2')
            conv3 = conv_relu_vgg(pool2, name='conv3_1')
            conv3 = conv_relu_vgg(conv3, name='conv3_2')
            conv3 = conv_relu_vgg(conv3, name='conv3_3')

            # 28 28
            #pool3 = tf.layers.max_pooling2d(conv3, 2, 2, name='pool3')
            pool3, max3 = max_pooling(conv3, 2, 2, name='pool3')
            conv4 = conv_relu_vgg(pool3, name='conv4_1')
            conv4 = conv_relu_vgg(conv4, name='conv4_2')
            conv4 = conv_relu_vgg(conv4, name='conv4_3')

            # 14 14
            #pool4 = tf.layers.max_pooling2d(conv4, 2, 2, name='pool4')
            pool4, max4 = max_pooling(conv4, 2, 2, name='pool4')
            conv5 = conv_relu_vgg(pool4, name='conv5_1')
            conv5 = conv_relu_vgg(conv5, name='conv5_2')
            conv5 = conv_relu_vgg(conv5, name='conv5_3')

            # 7 7
            pool5, max5 = max_pooling(conv5, name='pool5')
            conv6 = conv_relu(pool5, 4096, ksize=3, stride=1, name='conv6_1')
            conv6 = conv_relu(conv6, 1000, ksize=3, stride=1, name='conv6_2')
            conv6 = conv_relu(conv6, 512, ksize=3, stride=1, name='conv6_3')

            up1 = up_sample(conv6, max5)
            upconv1 = conv_relu(up1, 512, ksize=3, stride=1, name='upconv1_1')
            upconv1 = conv_relu(upconv1, 512, ksize=3, stride=1, name='upconv1_2')
            upconv1 = conv_relu(upconv1, 512, ksize=3, stride=1, name='upconv1_3')

            up2 = up_sample(upconv1, max4)
            upconv2 = conv_relu(up2, 256, ksize=3, stride=1, name='upconv2_1')
            upconv2 = conv_relu(upconv2, 256, ksize=3, stride=1, name='upconv2_2')
            upconv2 = conv_relu(upconv2, 256, ksize=3, stride=1, name='upconv2_3')
            
            up3 = up_sample(upconv2, max3)
            upconv3 = conv_relu(up3, 128, ksize=3, stride=1, name='upconv3_1')
            upconv3 = conv_relu(upconv3, 128, ksize=3, stride=1, name='upconv3_2')
            upconv3 = conv_relu(upconv3, 128, ksize=3, stride=1, name='upconv3_3')
            
            up4 = up_sample(upconv3, max2)
            upconv4 = conv_relu(up4, 64, ksize=3, stride=1, name='upconv4_1')
            upconv4 = conv_relu(upconv4, 64, ksize=3, stride=1, name='upconv4_2')
            
            up5 = up_sample(upconv4, max1)
            upconv5 = conv_relu(up5, 64, ksize=3, stride=1, name='upconv5_1')
            upconv5 = conv_relu(upconv5, 64, ksize=3, stride=1, name='upconv5_2')
            upconv5 = conv_relu(upconv5, self.num_class, ksize=3, stride=1, name='upconv5_3')

            out = tf.argmax(tf.nn.softmax(upconv5, dim=-1), axis=-1)

            if self.training is False:
                return out

            mask = tf.subtract(x, label_map['void'])
            mask = tf.reduce_sum(mask, -1)
            mask = tf.not_equal(mask, 0, name='mask')
            logits_masked = tf.boolean_mask(upconv5, mask)
            labels_masked = tf.boolean_mask(y, mask)

            cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=logits_masked,
                                                          labels=labels_masked)
            cross_entropy_sum = tf.reduce_mean(cross_entropies)
            
            return out,cross_entropy_sum

