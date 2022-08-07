from utils import data_process, model_fit, setting, random_seed
from utils.setting import log_string
import json
from models import deep_model
import torch

if __name__ == "__main__":


    config_name = 'config/model_config.json'
    args = setting.parser_set()

    alpha = args.alpha  # coefficient in loss function
    gamma = args.gamma  # switch training stage
    with open(config_name, 'r') as f:
        config = json.loads(f.read())
    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    random_seed.setup_seed(args.random_seed)

    log = open(config['log_path'] + 'model_name({})-alpha({})-gamma({})-seed({})'.format(args.model_name, alpha, gamma, args.random_seed) + '.txt', 'w')
    log_string(log, 'let us begin! ' + args.model_name + ' ○（*￣︶￣*）○\n')

    if args.model_name == 'FNN':
        model = deep_model.FNN(config['input_num'],
                               args.transmission_num,
                               config['output_num'],
                               config['time_step']).to(device)

    if args.model_name == 'RNN':
        model = deep_model.RNN(config['input_num'],
                               args.transmission_num,
                               config['output_num'],
                               config['time_step']).to(device)

    if args.model_name == 'GRU':
        model = deep_model.GRU(config['input_num'],
                               args.transmission_num,
                               config['output_num'],
                               config['time_step']).to(device)

    else:
        model = deep_model.LSTM(config['input_num'],
                                args.transmission_num,
                                config['output_num'],
                                config['time_step']).to(device)


    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['learning_rate'],
                                 eps=config['eps'])

    train_data, Z, R, max, min = data_process.load_data(config_name, log)
    model_fit.train(model, train_data, args.epochs, criterion, optimizer, device, gamma, alpha, log)

    # save model
    torch.save(model.state_dict(), 'save_model/model_name({})-alpha({})-gamma({})-seed({})'.format(args.model_name, alpha, gamma, args.random_seed))

    error, mean_cor, max_cor = model_fit.test(model, Z, R, max, min, device)

    log_string(log, 'results: mean_cor: {}, max_cor: {}, error: {}'.format(mean_cor, max_cor, error))
    log.close()

