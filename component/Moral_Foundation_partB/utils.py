long_mfs = ['care', 'harm', 'fairness', 'cheating', 'loyalty',
            'betrayal', 'authority', 'subversion', 'purity', 'degradation', 'non-moral']


def convert_cls(row, N):  # todo calculate for other levels
    import numpy as np
    row_abs = row.apply(np.abs)
    n_top = row_abs.nlargest(N)
    results = []
    for max_mf in n_top.index:
        max_mf_bias = row[max_mf]
        if 'care' in max_mf:
            if max_mf_bias > 0:
                results.append(0)
            else:
                results.append(1)
        elif 'fairness' in max_mf:
            if max_mf_bias > 0:
                results.append(2)
            else:
                results.append(3)
        elif 'loyalty' in max_mf:
            if max_mf_bias > 0:
                results.append(4)
            else:
                results.append(5)
        elif 'authority' in max_mf:
            if max_mf_bias > 0:
                results.append(6)
            else:
                results.append(7)
        elif 'purity' in max_mf:
            if max_mf_bias > 0:
                results.append(8)
            else:
                results.append(9)
    return results


def top_n_accuracy_single_col(n, scores, gold_labels):
    import numpy as np
    topn = scores.apply(lambda row: convert_cls(row, n), axis=1)
    # topn = np.argsort(converted, axis=1)[:, -n:]
    top_n_acc = np.mean(np.array([1 if gold_labels[k] in topn[k] else 0 for k in range(len(topn))]))
    return top_n_acc


def top_n_accuracy_expanded(n, scores, gold_labels):  # todo move to utils
    import numpy as np
    topn = np.argsort(scores, axis=1)[:, -n:]
    top_n_acc = np.mean(np.array([1 if gold_labels[k] in topn[k] else 0 for k in range(len(topn))]))
    return top_n_acc


def metrics_calc(outputs, targets, output_single_col):
    from sklearn import metrics
    import numpy as np
    if output_single_col:
        # for N in range(1, len(set(targets))):
        #     top_n = top_n_accuracy_single_col(n=N, scores=outputs, gold_labels=targets)
        #     print(f'Top-{N} accuracy:', top_n)
        accuracy = metrics.accuracy_score(targets, outputs)
        f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
        f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
        print(f'Accuracy Score = {accuracy}')
        print(f'F1 Score (Micro) = {f1_score_micro}')
        print(f'F1 Score (Macro) = {f1_score_macro}')
        cm = metrics.confusion_matrix(targets, outputs)
        print(cm)
        # todo make correlation right
    else:
        for N in range(1, outputs.shape[1]):
            top_n = top_n_accuracy_expanded(n=N, scores=outputs, gold_labels=targets)  # check the function for top_n
            print(f'Top-{N} accuracy:', top_n)
        arg_max = np.argmax(outputs, axis=1)
        accuracy = metrics.accuracy_score(targets, arg_max)
        f1_score_micro = metrics.f1_score(targets, arg_max, average='micro')
        f1_score_macro = metrics.f1_score(targets, arg_max, average='macro')
        print(f'Accuracy Score = {accuracy}')
        print(f'F1 Score (Micro) = {f1_score_micro}')
        print(f'F1 Score (Macro) = {f1_score_macro}')
        cm = metrics.confusion_matrix(targets, arg_max)
        print(cm)

        from scipy.stats import spearmanr
        # convert targets to one-hot
        n_values = targets.max() + 1
        one_hot_targets = np.eye(n_values)[targets]
        # Spearman’s Correlation
        corr_s, _ = spearmanr(outputs, one_hot_targets,
                              axis=0)  # axis=0 each column represents a variable, with observations in the rows.
        corr_s = np.diag(corr_s[n_values:, 0:n_values])
        print(f'Spearman’s Correlation:\n {corr_s}')
        # Pearson’s Correlation
        corr_p = np.corrcoef(outputs, one_hot_targets, rowvar=False)
        corr_p = np.diag(corr_p[n_values:, 0:n_values])
        # corr_p, _ = pearsonr(outputs, targets, axis=0)  # axis=0 each column represents a variable, with observations in the rows.
        print(f'Pearson’s Correlation: \n {corr_p}')


def infer_bert_model(testing_loader, bert_model):
    import numpy as np
    bert_model.eval()
    test_embeddings = []
    test_targets = []
    with torch.no_grad():
        for step, batch in enumerate(testing_loader):
            batch = [r.to(bert_model.device) for r in batch]
            if len(batch) == 4:
                ids, mask, token_type_ids, label = batch
                labels_in_batch = True
            elif len(batch) == 3:
                ids, mask, token_type_ids = batch
                labels_in_batch = False
            outputs = bert_model(ids, mask, token_type_ids)
            embeddings = outputs.last_hidden_state[:, 0, :]  # todo also try: outputs.pooler_output
            # big_val, big_idx = torch.max(outputs.data, dim=1)
            test_embeddings.append(embeddings.cpu().detach().numpy())
            if labels_in_batch:
                test_targets.append(label.cpu().detach().numpy())

    test_embeddings = np.concatenate(test_embeddings, axis=0)
    if labels_in_batch:
        test_targets = np.concatenate(test_targets, axis=0)
        return test_targets, test_embeddings

    return test_embeddings


def encode_texts(texts, output_hidden_states, model, tokenizer, shuffle):
    # from torch.utils.data import DataLoader
    import torch
    print('Starting encoding text...')
    # todo use fast tokenizer here
    tokenized_data = handle_tokenize(texts=texts, tokenizer=tokenizer)
    encoding_loader = torch.utils.data.DataLoader(tokenized_data,
                                                  batch_size=100)  # , shuffle=shuffle)  # , num_workers=4)
    embs = []
    text_embeddings = []
    for step, batch in enumerate(encoding_loader):
        if step % 10000 == 0:
            print('Step: ', step)
        batch = [r.to(model.device) for r in
                 batch]  # todo can't we do this before?
        ids, mask, token_type_ids = batch

        outputs = model(ids, mask, token_type_ids, output_hidden_states=output_hidden_states)
        # odict_keys(['last_hidden_state', 'pooler_output'])
        if output_hidden_states:
            # import pdb; pdb.set_trace()
            hidden_states = outputs.hidden_states
            text_embeddings.append(hidden_states[-1].detach().to('cpu').numpy())
        else:
            # import pdb; pdb.set_trace()
            last_hidden_states = outputs.last_hidden_state  # todo last_hidden state or pooled?
            # print(last_hidden_states.size())
            e = last_hidden_states[:, 0, :]  # todo try pooler_output
            # p = outputs.pooler_output
            # print(e.size())
            # inputs = inputs.detach().cpu()
            # print('e shape', e.shape)
            embs.append(e)  # embs.append(p)

        # for r in batch:
        #     r.to('cpu')
    if output_hidden_states:
        text_embeddings = np.concatenate(text_embeddings, axis=0)
        return text_embeddings, tokenized_data
    else:
        # import pdb; pdb.set_trace()
        embeddings = torch.vstack(embs)
        return embeddings


def read_bert_model(model_path):
    import torch
    import transformers
    model = transformers.BertModel.from_pretrained('bert-base-uncased')
    model.load_state_dict(torch.load(model_path))
    return model





# def plot_contour(df, labels, plot_name):
#     import seaborn as sns
#     # choose a color palette with seaborn.
#     # palette = np.array(sns.color_palette("hls", num_classes))
#
#     # create a scatter plot.
#     f = plt.figure(figsize=(20, 20))
#     # ax = plt.subplot(aspect='equal')
#     sns.kdeplot(x=df[:, 0], y=df[:, 1], hue=labels, legend='full', alpha=0.5, fill=False)
#     # plt.xlim(-25, 25)
#     # plt.ylim(-25, 25)
#     # ax.axis('off')
#     # ax.axis('tight')
#     plt.savefig(f'./mul_class/plots/contour_{plot_name}.png')



def handle_tokenize(texts, tokenizer, labels=None):
    from torch.utils.data import TensorDataset
    import torch
    print('starting tokenizing...')
    encoding = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    # dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
    ids = encoding['input_ids']  # default max_seq 512
    mask = encoding['attention_mask']
    token_type_ids = encoding['token_type_ids']

    if labels:
        targets = torch.tensor(labels)
        print('seq, mask and labels are ready')
        return TensorDataset(ids, mask, token_type_ids, targets)
    else:
        print('seq, mask are ready')
        return TensorDataset(ids, mask, token_type_ids)


def reduce_dim(embs, pca=True):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    if pca:
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embs)
    else:
        tsne = TSNE(n_components=2, verbose=1, perplexity=400, n_iter=1000)
        reduced = tsne.fit_transform(embs)
    return reduced


# def plot_scatter(components, pca_virtue, pca_vice, sentiments):
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     i = 0
#     j = 1
#     vice_x, vice_y = (pca_vice[0, i], pca_vice[0, j])
#     virtue_x, virtue_y = (pca_virtue[0, i], pca_virtue[0, j])
#
#     sns.scatterplot(x=components[:, i], y=components[:, j], hue=sentiments, alpha=0.3, s=5)
#     plt.arrow(vice_x, vice_y, virtue_x, virtue_y, head_width=1, head_length=1, fc='k', ec='k')

def plot_scatter(x, y, labels, markers, plot_file):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.color_palette("Paired")  # todo doesn't work
    # choose a color palette with seaborn.
    # palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    plt.figure(figsize=(12, 12))
    # ax = plt.subplot(aspect='equal')
    markers_map = {'vice': 'v', 'virtue': 'P', 'non-moral': 'o'}
    # todo use matplotlip or plotly (with showing text when moving curser)
    sns.scatterplot(x=x, y=y, hue=labels, style=markers, legend='full', alpha=0.5, markers=markers_map)
    # plt.xlim(-25, 25)
    # plt.ylim(-25, 25)
    # ax.axis('off')
    # ax.axis('tight')
    # plt.savefig(f'./plots/scatter_{plot_name}.png')
    plt.savefig(plot_file)


def plot_contour(df, labels, plot_name):
    # choose a color palette with seaborn.
    # palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(20, 20))
    # ax = plt.subplot(aspect='equal')
    sns.kdeplot(x=df[:, 0], y=df[:, 1], hue=labels, legend='full', alpha=0.3)
    # plt.xlim(-25, 25)
    # plt.ylim(-25, 25)
    # ax.axis('off')
    # ax.axis('tight')
    plt.savefig(f'./plots/contour_{plot_name}.png')


def convert_labels(numeric_labels, origin_level, target_level):
    import numpy as np
    # if origin_level == target_level:
    #     return numeric_labels

    if origin_level not in ['all_fine_grained', 'sentiment']:
        raise ValueError(f'Unrecognized origin label level {origin_level}')
    if origin_level == 'all_fine_grained':
        maps = {}
        maps['all_fine_grained'] = {0: 'care', 1: 'harm', 2: 'fairness', 3: 'cheating', 4: 'loyalty', 5: 'betrayal',
                                    6: 'authority', 7: 'subversion', 8: 'purity', 9: 'degradation', 10: 'non-moral'}
        maps['category'] = {0: 'care', 1: 'care', 2: 'fairness', 3: 'fairness', 4: 'loyalty', 5: 'loyalty',
                            6: 'authority', 7: 'authority', 8: 'purity', 9: 'purity', 10: 'non-moral'}
        maps['sentiment'] = {0: 'virtue', 1: 'vice', 2: 'virtue', 3: 'vice', 4: 'virtue', 5: 'vice',
                             6: 'virtue', 7: 'vice', 8: 'virtue', 9: 'vice', 10: 'non-moral'}
        target_map = maps[target_level]
        converted = np.vectorize(target_map.get)(numeric_labels)
    elif origin_level == 'sentiment':
        if target_level != 'sentiment':
            # todo warning here
            print(f'for origin label level sentiment, we can only convert to sentiment')
        target_map = {0: 'virtue', 1: 'vice', 2: 'non-moral'}
        converted = np.vectorize(target_map.get)(numeric_labels)
    return converted


def plot_features(X, y, origin_level, plot_name, use_pca):
    print('************************************')
    # reduce shape
    print(X.shape)
    if use_pca == "full":
        reduced = X[['pc1_cls', 'pc2_cls']]
        reduced = reduced.values
    else:
        reduced = reduce_dim(embs=X, pca=use_pca)
    print('reduced shape: ', reduced.shape)

    y_converted = {}
    for l in ['all_fine_grained', 'category', 'sentiment']:
        print(f'converting labels from {origin_level} level to {l} level')
        y_converted[l] = convert_labels(y, origin_level=origin_level, target_level=l)
        # import pdb; pdb.set_trace()

    print('plotting the features')
    # plot_name_l = plot_name.replace('XXX', l)
    print(f'saving plot to {plot_name}')
    plot_scatter(x=reduced[:, 0], y=reduced[:, 1], labels=y_converted['category'], markers=y_converted['sentiment'],
                 plot_file=plot_name)
    # plot_contour(reduced, targets, plot_name)
    print('************************************')


def load_splits(label_level, mode, read_data_name=None):
    def remove_non_moral(df):
        if label_level == 'category':
            removed_df = df[df['MF'] != 5]
        elif label_level == 'sentiment':
            removed_df = df[df['MF'] != 2]
        elif label_level == 'all_fine_grained':
            removed_df = df[df['MF'] != 10]
        else:
            raise ValueError(f'Unrecognized label level {label_level}')
        return removed_df

    import pandas as pd
    if read_data_name == 'vignettes':
        fetched_df = pd.read_csv(f'./data/vignettes.csv')
        texts = fetched_df.text.tolist()
        labels = fetched_df.polarity.replace({'positive': 0, 'negative': 1})
        labels = labels.tolist()
        num_class = 2
        return texts, labels, num_class

    elif (read_data_name == 'twitter10') or (read_data_name == 'twitter11'):
        file_name = 'twitter'
        fetched_df = pd.read_csv(f'./data/{file_name}_{mode}_{label_level}.csv')
        if read_data_name == 'twitter10':
            fetched_df = remove_non_moral(df=fetched_df)

    elif read_data_name == 'emfd':
        file_name = 'emfd'
        fetched_df = pd.read_csv(f'./data/emfd_{mode}_{label_level}.csv')
    else:
        raise ValueError(f'Unrecognized data name {read_data_name}')

    fetched_df = fetched_df.dropna().reset_index(drop=True)
    texts, labels = fetched_df['prep_text'].tolist(), fetched_df['MF'].tolist()
    num_class = len(set(labels))

    print(f'Reading data: {read_data_name} in mode: {mode}')
    print(f'{mode} data shape: ', len(texts), f', {mode} labels shape: ', len(labels))
    assert len(texts) == len(labels), f'number of {mode} texts and {mode} labels are not equal'
    print('num classes: ', num_class)

    if mode == "train":
        val_df = pd.read_csv(f'./data/{file_name}_val_{label_level}.csv')
        if read_data_name == 'twitter10':
            val_df = remove_non_moral(df=val_df)

        val_df = val_df.dropna().reset_index(drop=True)
        val_texts, val_labels = val_df['prep_text'].tolist(), val_df['MF'].tolist()
        # , max_len=MAX_LEN)

        # print('train data shape: ', len(texts), ', train labels shape: ', len(labels))
        print('val data shape: ', len(val_texts), ', val labels shape: ', len(val_labels))
        assert len(val_texts) == len(val_labels), 'number of val texts and val labels are not equal'
        return texts, labels, val_texts, val_labels, num_class
    else:
        return texts, labels, num_class


def load_fine_tuned_results(label_level, mode, train_data_name=None, test_data_name=None):
    import pandas as pd
    read_data_name = train_data_name if mode == "train" else test_data_name
    long_mfs = ['care', 'harm', 'fairness', 'cheating', 'loyalty',
                'betrayal', 'authority', 'subversion', 'purity', 'degradation', 'non-moral']
    fetched_data = pd.read_csv(
        f'./fine_tuned_results/train_{train_data_name}_train_level_{label_level}_test_{test_data_name}_mode_{mode}.csv')
    print(f'Reading data: {read_data_name} in mode: {mode}')
    if test_data_name == "twitter" and train_data_name == 'twitter':
        mf_cols = long_mfs
    else:
        mf_cols = long_mfs[:-1]

    predicted_mfs = fetched_data[mf_cols].values
    labels = fetched_data.ground_truth.values
    num_class = len(set(labels))
    print('num classes: ', num_class)
    return predicted_mfs, labels, num_class


import torch


class BERTClass(torch.nn.Module):

    def __init__(self, num_classes):
        import transformers
        import torch
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')  # todo also try roberta
        # self.l2 = torch.nn.Dropout(0.3)
        # self.l3 = torch.nn.Linear(768, num_labels)
        # self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.5)
        self.classifier = torch.nn.Linear(768, num_classes)

    def forward(self, ids, mask, token_type_ids):
        output_1 = self.l1(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
        # pooler = self.pre_classifier(output_1.pooler_output)
        # pooler = torch.nn.ReLU(output_1.pooler_output)#(pooler)  # todo why???
        pooler = self.dropout(output_1.pooler_output)
        output = self.classifier(pooler)
        return output, output_1.last_hidden_state[:, 0, :]

    def save_bert(self, save_path):
        import torch
        torch.save(self.l1.state_dict(), save_path)
        # self.l1.save_pretrained(save_path, push_to_hub=True, repo_name='my-awesome-model')

    def save_model(self, save_path):
        import pickle
        with open(save_path, 'wb') as file:
            pickle.dump(self, file)
