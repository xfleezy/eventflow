import argparse

def configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--event_path',
                        type=str,
                        help="Path to saved model.",
                        default="event")
    parser.add_argument('--gray_scale_path',
                        type=str,
                        help="Specific saved model to load. A new one will be generated if empty.",
                        default="gray")
    parser.add_argument('--save_path',
                        type=str,
                        help="Path to log summaries.",
                        default='../data/model')
    parser.add_argument('--batch_size',
                        type=int,
                        help="Training batch size.",
                        default=16)
    parser.add_argument('--lr',
                        type=float,
                        help="Initial learning rate.",
                        default=1e-4)
    parser.add_argument('--weight_decay',
                        type=float,
                        help="Rate at which the learning rate is decayed.",
                        default=1e-4)
    parser.add_argument('--gamma',
                        type=float,
                        help='Rate at which the learning rate is decayed.',
                        default=0.5)
    parser.add_argument('--n_iter',
                        type=float,
                        help='Rate at which print loss.',
                        default=50)
    parser.add_argument('--milestone',
                        type=float,
                        help='epoch where the learning rate is decayed.',
                        default=[15,30,50])
    parser.add_argument('--size',
                        type=float,
                        help='image resolution for each pyramid level',
                        default= [(344, 256), (172, 128), (86, 64)])
    parser.add_argument('--epoch',
                        type=float,
                        help='epoch where the learning rate is decayed.',
                        default=80)
    parser.add_argument('--smoothness_weight',
                        type=float,
                        help='Weight for the smoothness term in the loss function.',
                        default=0.5)
    parser.add_argument('--count_only',
                        action='store_true',
                        help='If true, inputs will consist of the event counts only.')
    parser.add_argument('--time_only',
                        action='store_true',
                        help='If true, inputs will consist of the latest timestamp only.')

    # Args for testing only.
    parser.add_argument('--test_sequence',
                        type=str,
                        help="Name of the test sequence.",
                        default='outdoor_day1')
    parser.add_argument('--gt_path',
                        type=str,
                        help='Path to optical flow ground truth npz file.',
                        default='')
    parser.add_argument('--test_plot',
                        action='store_true',
                        help='If true, the flow predictions will be visualized during testing.')
    parser.add_argument('--test_skip_frames',
                        action='store_true',
                        help='If true, input images will be 4 frames apart.')
    parser.add_argument('--save_test_output',
                        action='store_true',
                        help='If true, output flow will be saved to a npz file.')

    args = parser.parse_args()
    return args
