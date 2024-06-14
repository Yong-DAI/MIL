import torch
import os
import argparse
from datasets.crowd import Crowd
from models.fusion import fusion_model
from utils.evaluation import eval_game, eval_relative
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--data-dir', default='./datasets',
                        help='training data directory')
parser.add_argument('--save-dir', default='./savemodels/0321-163709-rgb-false-1e-7v2',
                        help='model directory')
parser.add_argument('--model', default='best_model_10.pth'
                    , help='model name')
parser.add_argument('--crop-size', type=int, default=224,
                        help='default 256')
parser.add_argument('--downsample-ratio', type=int, default=8,                    
                        help='downsample ratio')
parser.add_argument('--device', default='0', help='gpu device')
args = parser.parse_args()

if __name__ == '__main__':


#     datasets = Crowd(os.path.join(args.data_dir, 'test'), method='test')
#     dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
#                                              num_workers=8, pin_memory=True)
    

    datasets =  Crowd(os.path.join(args.data_dir, 'test'),
                                  args.crop_size,
                                  args.downsample_ratio,
                                  'test') 
    dataloader = DataLoader(datasets,
                                          collate_fn=default_collate,
                                          batch_size= 1,
                                          shuffle= False,
                                          num_workers=8,
                                          pin_memory=False)
                           
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
    device = torch.device('cuda')

    model = fusion_model()
    model.to(device)
    model_path = os.path.join(args.save_dir, args.model)
    checkpoint = torch.load(model_path, device)
    model.load_state_dict(checkpoint)
    print('trained checkpoint loaded')
    model.eval()

    print('testing...')
    # Iterate over data.
    game = [0, 0, 0, 0]
    mse = [0, 0, 0, 0]
    mae = [0, 0, 0, 0]
    total_relative_error = 0

    for inputs, target, name in tqdm(dataloader):
        if type(inputs) == list:
            inputs[0] = inputs[0].to(device)
            inputs[1] = inputs[1].to(device)
        else:
            inputs = inputs.to(device)

        # inputs are images with different sizes
        if type(inputs) == list:
            assert inputs[0].size(0) == 1
        else:
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
        with torch.set_grad_enabled(False):
            outputs,_,_ = model(inputs)

            for L in range(4):
                abs_error, square_error = eval_game(outputs, target, L)
                game[L] += abs_error
                mse[L] += square_error
                mae[L] += torch.sqrt(square_error)
            relative_error = eval_relative(outputs, target)
            total_relative_error += relative_error

    N = len(dataloader)
    game = [m / N for m in game]
    mse = [torch.sqrt(m / N) for m in mse]
    total_relative_error = total_relative_error / N
    
    MAE = [(m / N) for m in mae]

    log_str = 'Test{}, GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} ' \
              'RMSE {mse:.2f} mae {mae:.4f} Re {relative:.4f}, '.\
        format(N, game0=game[0], game1=game[1], game2=game[2], game3=game[3], mse=mse[0],  mae=MAE[0] , relative=total_relative_error)

    print(log_str)

