from util import *
from SegNetModel import *
#default resolution is 640*480
def main(config):

    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    path = path_pack(config)
    path.check_path()
    img_height = config.resolution[-1]
    img_width = config.resolution[0]
    SegNet = SegNetModel(config, img_width, img_height)
    t0 = time.time()
    if config.training:
        print('\nLoading data from '+path.data_dir)

        fn_img = ReadTextFile(path.data_dir+'/train.txt', path.data_dir+'/images', 'jpg')          
        fn_seg = ReadTextFile(path.data_dir+'/train.txt', path.data_dir+'/labels', 'png')

        SegNet.SegNet_training_setup(fn_seg, fn_img)
        
    else:
        print('\nLoading data from '+path.data_dir)
        
        fn_img = sorted(ReadTextFile(path.data_dir+'/ImageSets/Segmentation/train.txt',
                                    path.data_dir+'/JPEGImages', 'jpg'), key=numericalSort)
        fn_seg = sorted(ReadTextFile(path.data_dir+'/ImageSets/Segmentation/train.txt',
                                path.data_dir+'/SegmentationClass', 'png'), key=numericalSort)

        SegNet.SegNet_testing_setup(fn_seg, fn_img)

    print('Finished loading in %.2f seconds.' % (time.time() - t0))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        SegNet.chk_point_restore(sess)

        if config.training:
            SegNet.SegNet_training(sess, path)
        else:
            SegNet.SegNet_testing(sess, path, fn_img)
            


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SegNet_demo')

    parser.add_argument('--train', dest='training', help='set train_flag, default is True',
                        default=True, type=str2bool)
    parser.add_argument('--batch_size', dest='batch_size', help='Number of images in each batch',
                        default=4, type=int)
    parser.add_argument('--num_epoch', dest='num_epoch', help='Total number of epochs to run for training',
                        default=100000, type=int)
    parser.add_argument('--init_learning_rate', dest='init_learning_rate', help='Initial learning rate',
                        default=1e-5, type=float)
    parser.add_argument('--learning_rate_decay', dest='learning_rate_decay', help='Ratio for decaying the learning rate after each epoch',
                        default=0.95, type=float)
    parser.add_argument('--gpu', dest='gpu', help='GPU to be used',
                        default='0', type=str)
    parser.add_argument('--threshold', dest='threshold', help='threshold to display',
                        default=0.9, type=float)
    parser.add_argument('--saved_name', dest='saved_name', help='image saved name for data collection',
                        default='table_9_', type=str)
    parser.add_argument('--label', dest='label', help='object label for data collection',
                        default='table', type=str)
    parser.add_argument('--dataset', dest='dataset', help='dataset name to save',
                        default='progress', type=str)
    parser.add_argument('--train_test_dataset', dest='train_test_dataset', help='train and test dataset directory',
                        default='./data/train', type=str)
    parser.add_argument('--oneshot_img', dest='oneshot_img',  nargs='+', help='oneshot image training name',
                        default=['001'], type=str)
    parser.add_argument('--resolution', dest='resolution', nargs='+', help='image resolution, [width height]',
                        default = [224, 224], type=int)
    config = parser.parse_args()

    return config

if __name__ == '__main__':
    config = parse_args()
    main(config)

    




