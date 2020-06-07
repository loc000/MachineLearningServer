import argparse
from core.utils import load_coco_data


def main(use_inception):
    # load train dataset
    print("Loading COCO training data...")
    data = load_coco_data(data_path='./data', split='train')
    print("Done!")
    word_to_idx = data['word_to_idx']
    # load val dataset to print out bleu scores every epoch
    val_data = load_coco_data(data_path='./data', split='val')

    if use_inception:
        L = 64
        D = 2048
    else:
        L = 196
        D = 512

    from core.solver import CaptioningSolver
    from core.model import CaptionGenerator
    model = CaptionGenerator(word_to_idx, dim_feature=[L, D], dim_embed=512,
                                       dim_hidden=1800, n_time_step=16, prev2out=True, 
                                                 ctx2out=True, alpha_c=5.0, selector=True, dropout=True)

    solver = CaptioningSolver(model, data, val_data, n_epochs=100, batch_size=256, update_rule='adam',
                                          learning_rate=0.0005, print_every=1000, summary_every=10000, save_every=1, image_path='./image/',
                                    pretrained_model=None, model_path='model/lstm/', test_model='model/lstm/model-10',
                                     print_bleu=True, log_path='log/')

    solver.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_inception", action="store_true", help="use inception network image size (299 * 299)")
    args = parser.parse_args()
    main(args.use_inception)

