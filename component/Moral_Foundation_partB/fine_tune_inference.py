import torch
from torch import cuda
from torch.utils.data import DataLoader
from transformers import BertTokenizer
# from transformers.models.bert.modeling_bert import BertModel
from utils import handle_tokenize, long_mfs
import pickle

def validation(val_loader, model):
    import numpy as np
    model.eval()
    fin_outputs = []
    fin_embeddings = []
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            batch = [r.to(device) for r in batch]

            if len(batch) == 4:
                ids, mask, token_type_ids, label = batch
                labels_in_batch = True
            elif len(batch) == 3:
                ids, mask, token_type_ids = batch
                labels_in_batch = False

            outputs, embeddings = model(ids, mask, token_type_ids)

            # big_val, big_idx = torch.max(outputs.data, dim=1)
            fin_outputs.append(outputs.cpu().detach().numpy())
            fin_embeddings.append(embeddings.cpu().detach().numpy())
    fin_outputs = np.concatenate(fin_outputs, axis=0)
    fin_embeddings = np.concatenate(fin_embeddings, axis=0)
    return fin_outputs, fin_embeddings


if __name__ == '__main__':
    device = 'cuda' if cuda.is_available() else 'cpu'
    print('device is ', device)
    # Sections of config
    MAX_LEN = 200
    TRAIN_BATCH_SIZE = 100
    VALID_BATCH_SIZE = 4
    EPOCHS = 10
    LEARNING_RATE = 1e-05
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # make_twitter_splits_levels()
    # make_emfd_splits_levels()
    label_level = 'all_fine_grained'
    fine_tuning_data = ('twitter', 'twitter')
    file_name = f'train_{fine_tuning_data[0]}_test_{fine_tuning_data[1]}'
    mf_cols = long_mfs

    '''read saved model'''
    model_path = 'mul_class_all_fine_grained_train_twitter_test_twitter_maxe_15_b100.pkl'
    with open(model_path, 'rb') as file:

        model = pickle.load(file)
    model.to(device)
    # print(f'model device: {model.device}')

    print('~~~~~~~~~~~~~~~ Starting Testing ~~~~~~~~~~~~~~~')
    '''read test data'''
    test_texts = ["hello how are you","I am fine"]
    test_set = handle_tokenize(texts=test_texts, tokenizer=tokenizer)
    testing_loader = DataLoader(test_set, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=4)

    MF_outputs, _ = validation(testing_loader, model)

    Final_list = []

    for output in MF_outputs:
        temp = dict(zip(long_mfs,output))
        Final_list.append(temp)

    print(Final_list)
