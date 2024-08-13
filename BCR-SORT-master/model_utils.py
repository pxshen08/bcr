import os
import logging
import pandas as pd
import numpy as np
import torch

from tqdm import tqdm
from functools import partial
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path
from modelnojv import load_model
from data_loader import load_dataset
from data_utils import sequence_padding, generate_dict, decode_sequence, decode_categorical, generate_seq_mask,generate_seq_mask1
import umap
import matplotlib.pyplot as plt

def predict(args):
    file_in = args.input_file
    file_out = args.output_file
    if file_out is None:
        file_out = file_in.replace('.csv', '_bcr-sort.csv')

    model_path = args.model_path
    device = args.device
    if device is not None:
        torch.cuda.set_device(device)
        print("Use GPU: {} for loading model".format(torch.cuda.current_device()))
    else:
        device = torch.device('cpu')
    model= args.model
    # load data
    DatasetBCR = load_dataset(file_in, mode='predict',model=model)
    sequence_padder = partial(sequence_padding, mode='predict')
    DataLoaderBCR = DataLoader(DatasetBCR, batch_size=1024, num_workers=0, pin_memory=True, drop_last=False,
                               shuffle=False, collate_fn=sequence_padder)

    # load model
    model = load_model()
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    model.to(device)
    model.eval()

    # setup output vars
    encoding_dict = generate_dict()
    ntest = len(DataLoaderBCR.dataset)
    num_class = 3
    pred = np.zeros((ntest, num_class))
    col_list = ['seq', 'V_gene', 'J_gene', 'isotype', 'pred', 'p_naive', 'p_ASC', 'p_memory']
    data_out = {}
    for col in col_list:
        data_out[col] = []

    all_encoder_outputs = []
    all_outputs = []

    # prediction
    dim_start = 0
    for i, dataloader in enumerate(DataLoaderBCR, 0):
        inputs, lengths, _ = dataloader
        sequence, vgene, jgene, isotype = inputs
        sequence = sequence.type(torch.cuda.LongTensor).to(device)
        vgene = vgene.type(torch.cuda.LongTensor).to(device)
        jgene = jgene.type(torch.cuda.LongTensor).to(device)
        isotype = isotype.type(torch.cuda.LongTensor).to(device)

        outputs, _, encoder_output, outputs_lstm_conv = model(sequence, vgene, jgene, isotype, lengths, return_aux_out=True)

        # all_encoder_outputs.append(encoder_output.cpu().detach().numpy())
        all_encoder_outputs.append(outputs_lstm_conv.cpu().detach().numpy())
        all_outputs.append(outputs.cpu().detach().numpy())

        # collect outputs
        dim_end = dim_start + outputs.shape[0]
        # pred[dim_start:dim_end, ] = outputs.cpu().detach().numpy()
        outputs_np = outputs.cpu().detach().numpy()

        # 确保 outputs_np 的形状与 pred 的形状匹配
        if outputs_np.shape[1] != pred.shape[1]:
            # 处理形状不匹配的问题，可以考虑扩展或截断数组
            # 例如，这里假设我们只需要前两列
            outputs_np = outputs_np[:, :pred.shape[1]]

        pred[dim_start:dim_end, ] = outputs_np

        sequence_out = decode_sequence(sequence.cpu().detach().tolist())
        vgene = vgene.squeeze().cpu().detach().tolist()
        jgene = jgene.squeeze().cpu().detach().tolist()
        isotype = isotype.squeeze().cpu().detach().tolist()

        if not isinstance(vgene, list):
            vgene = [vgene]
            jgene = [jgene]
            isotype = [isotype]

        vgene_out = [decode_categorical(f, encoding_dict['V_gene']) for f in vgene]
        jgene_out = [decode_categorical(f, encoding_dict['J_gene']) for f in jgene]
        isotype_out = [decode_categorical(f, encoding_dict['isotype']) for f in isotype]

        data_out['seq'].extend(sequence_out)
        data_out['V_gene'].extend(vgene_out)
        data_out['J_gene'].extend(jgene_out)
        data_out['isotype'].extend(isotype_out)
        dim_start = dim_end

    # Save encoder outputs to a .pt file
    all_encoder_outputs = np.concatenate(all_encoder_outputs, axis=0)
    torch.save(torch.tensor(all_encoder_outputs), file_out.replace('.csv', '_encoder_outputslstm0729.pt'))
    all_outputs = np.concatenate(all_outputs, axis=0)
    torch.save(torch.tensor(all_outputs), file_out.replace('.csv', '_all_outputs0729.pt'))

    # save outputs
    pred_out = np.argmax(pred, axis=1).tolist()
    data_out['pred'] = [decode_categorical(pred, encoding_dict['label']) for pred in pred_out]
    data_out['p_naive'] = pred[:, 0].tolist()
    data_out['p_ASC'] = pred[:, 1].tolist()
    data_out['p_memory'] = pred[:, 2].tolist()

    Path(file_out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data_out).to_csv(file_out, index=False)

    # # UMAP embedding
    # reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='correlation')
    # embedding = reducer.fit_transform(all_encoder_outputs)
    # embedding1 = reducer.fit_transform(all_outputs)

    # # Visualization
    # plt.figure(figsize=(10, 8))
    # plt.scatter(embedding[:, 0], embedding[:, 1], c=pred_out, cmap='Spectral', s=1)
    # plt.colorbar()
    # plt.title('UMAP projection of BCRSORT encoder output')
    # plt.xlabel('UMAP1')
    # plt.ylabel('UMAP2')
    # plt.savefig('/home/mist/projects/Wang2023/scripts/BCR-SORT-master/2_umap0717.png')
    # plt.show()
    # plt.figure(figsize=(10, 8))
    # plt.scatter(embedding1[:, 0], embedding1[:, 1], c=pred_out, cmap='Spectral', s=1)
    # plt.colorbar()
    # plt.title('UMAP projection of BCRSORT encoder output')
    # plt.xlabel('UMAP1')
    # plt.ylabel('UMAP2')
    # plt.savefig('/home/mist/projects/Wang2023/scripts/BCR-SORT-master/3_umap0717.png')
    # plt.show()


def print_dataset_sample(loader, phase):
    data_iter = iter(loader)
    for _ in range(3):  # 打印3个样本
        inputs, lengths, labels = next(data_iter)
        print(f"Phase: {phase}, Sample input: {inputs[0]}, Label: {labels[0]}")

def train(args):
    """

    :param dataset_dict:
    :param param_input:
    :param model:
    :param dir_out:
    :param seed:
    :param device:
    :param include_status:
    :return:
    """

    # set up vars
    file_in = args.input_file
    dir_out = args.output_dir
    if dir_out is None:
        dir_out = os.path.join(os.path.dirname(file_in), 'train')

    device = args.device
    if device is not None:
        torch.cuda.set_device(device)
        print("Use GPU: {} for loading model".format(torch.cuda.current_device()))

    batch_size = args.batch_size #1024
    learning_rate = args.learning_rate #0.001
    weight_decay = args.weight_decay #0.05
    scaling_loss_aux = args.loss_scaling # 0.05
    num_epoch = args.num_epoch #10

    # file out
    Path(dir_out).mkdir(parents=True, exist_ok=True)
    file_wt = os.path.join(dir_out, "best_wt.pt")
    file_loss = os.path.join(dir_out, 'loss.csv')
    file_acc = os.path.join(dir_out, 'accuracy.csv')
    file_log = os.path.join(dir_out, 'log.log')
    logging.basicConfig(filename=file_log, level=logging.INFO)
    print("Project log directory: %s" % file_log)

    # load data
    model=args.model
    TrainDatasetBCR, ValDatasetBCR, TestDatasetBCR = load_dataset(file_in, mode='train', model=model,test_ratio=0.1)
    sequence_padder = partial(sequence_padding, mode='train')
    TrainDataLoaderBCR = DataLoader(TrainDatasetBCR, batch_size=batch_size, num_workers=0, pin_memory=True,
                                    drop_last=True, shuffle=True, collate_fn=sequence_padder)
    ValDataLoaderBCR = DataLoader(ValDatasetBCR, batch_size=batch_size, num_workers=0, pin_memory=True,
                                  drop_last=True, shuffle=True, collate_fn=sequence_padder)
    TestDataLoaderBCR = DataLoader(TestDatasetBCR, batch_size=batch_size, num_workers=0, pin_memory=True,
                                   drop_last=False, shuffle=False, collate_fn=sequence_padder)

    # load model
    model = load_model()
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer_ft = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer_ft, T_0=10)

    phase_list = ['train', 'val', 'test']
    phase_loader = {'train': TrainDataLoaderBCR, 'val': ValDataLoaderBCR, 'test': TestDataLoaderBCR}
    epoch_loss_log = {'train': [], 'val': [], 'test': []}
    epoch_accuracy = {'train': [], 'val': [], 'test': []}
    best_acc_test = 0.0
    with tqdm(total=num_epoch) as pbar:
        for epoch in range(0, num_epoch):

            for phase in phase_list:

                # print_dataset_sample(phase_loader[phase], phase)  # 在每个阶段打印样本

                if phase == 'train':
                    model.train()
                    grad_flag = True
                else:
                    model.eval()
                    grad_flag = False

                with torch.set_grad_enabled(grad_flag):

                    epoch_loss = 0.0
                    epoch_total = 0
                    epoch_correct = 0
                    batch_counter = 0
                    # Debug: Print phase_loader and phase information
                    print(f"Phase: {phase}")
                    print(f"Phase Loader Keys: {phase_loader.keys()}")
                    print(f"Length of DataLoader: {len(phase_loader[phase])}")

                    # Check if the DataLoader is empty
                    if len(phase_loader[phase]) < 0:
                        print(f"No data available for phase: {phase}")
                    else:
                        for i, dataloader in enumerate(phase_loader[phase], 0):
                            optimizer_ft.zero_grad()
                            inputs, lengths, labels = dataloader
                            sequence, vgene, jgene, isotype = inputs #sequence 1024,31
                            # sequence = sequence.type(torch.cuda.LongTensor).to(device)
                            vgene = vgene.type(torch.cuda.LongTensor).to(device)
                            jgene = jgene.type(torch.cuda.LongTensor).to(device)
                            isotype = isotype.type(torch.cuda.LongTensor).to(device)
                            labels = labels.to(device)

                            if phase == 'train':
                                if args.model=="antiBERTy":
                                   sequence1, labels_aux = generate_seq_mask1(sequence, lengths)
                                   labels_aux = labels_aux.to(device)
                                else:
                                    sequence_masked, labels_aux = generate_seq_mask(sequence, lengths)
                                    sequence = sequence_masked.to(device)  # 1024,32
                                    labels_aux = labels_aux.to(device)  # 1024

                                outputs, outputs_aux, _, ouput_lstm = model(sequence1, vgene, jgene, isotype, lengths, True) #outputs 1024,2 output_aux1024,22
                                loss = criterion(outputs, torch.argmax(labels, dim=1))# 1024,2
                            else:
                                outputs, outputs_aux, _, ouput_lstm = model(sequence, vgene, jgene, isotype, lengths,
                                                                            True)  # outputs 1024,2 output_aux1024,22
                                loss = criterion(outputs, torch.argmax(labels, dim=1))  # 1024,2

                            if phase == 'train':
                                loss_aux = criterion(outputs_aux.cuda(),labels_aux )
                                loss_total = loss + scaling_loss_aux * loss_aux

                                loss_total.backward()
                                optimizer_ft.step()
                                scheduler.step()

                        epoch_loss += loss.item()
                        epoch_total += outputs.shape[0]
                        pred = torch.argmax(outputs, dim=1)
                        epoch_correct += torch.sum(pred == torch.argmax(labels, dim=1))
                        print(batch_counter)
                        batch_counter += 1

                        pbar.set_description("epoch: %d, mode: %s, batch: %s, loss: %s, acc:%s" %
                                             (epoch + 1, phase,str(batch_counter), str(loss.cpu().detach().item()),
                                              str((epoch_correct / epoch_total).cpu().detach().item())))
                print(batch_counter)
                if batch_counter > 0:
                   loss_log = epoch_loss / batch_counter
                   epoch_loss_log[phase].append(loss_log)

                   acc = (epoch_correct / epoch_total).cpu().detach().item()
                   epoch_accuracy[phase].append(acc)
                   logging.info('[Epoch: %d, Mode: %s, Total: %d Accuracy: %.4f' % (epoch + 1, phase, epoch_total, acc))

                if phase == 'test' and acc > best_acc_test:
                    best_acc_test = acc
                    best_wt = model.state_dict()
                    torch.save(best_wt, file_wt)

            pbar.update()

    data_loss = pd.DataFrame({'train': epoch_loss_log['train'], 'val': epoch_loss_log['val'], 'test': epoch_loss_log['test']})
    # data_loss = pd.DataFrame(
    #     {'train': epoch_loss_log['train']})
    data_loss.groupby(level=0).max().to_csv(file_loss, index_label='epoch')
    # data_acc = pd.DataFrame({'train': epoch_accuracy['train']})
    data_acc = pd.DataFrame(
        {'train': epoch_accuracy['train'], 'val': epoch_accuracy['val'], 'test': epoch_accuracy['test']})
    data_acc.groupby(level=0).max().to_csv(file_acc, index_label='epoch')

