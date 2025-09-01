import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy, accuracy_single
from scipy.spatial.distance import cdist
import time

EPSILON = 1e-8
batch_size = 64

class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 2

        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]
        self.args = args

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(
            self._targets_memory
        ), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim
    
    def build_rehearsal_memory(self, data_manager, per_class):
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def tsne(self, showcenters=False, Normalize=False):
        import umap
        import matplotlib.pyplot as plt
        print('now draw tsne results of extracted features.')
        class_name = ['person', 'guitar', 'tv_stand', 'toilet', 'glass_box', 'monitor', 'bathtub', 'door', 'cup', 'sofa', 'tent', 'mantel', 'bed', 'piano', 'radio', 'chair', 'desk', 'bowl', 'flower_pot', 'laptop', 'plant', 'dresser', 'range_hood', 'stool', 'night_stand', 'car', 'lamp', 'curtain', 'bottle', 'airplane', 'wardrobe', 'keyboard', 'bookshelf', 'cone', 'vase', 'stairs', 'bench', 'sink', 'xbox', 'table']
        tot_classes=self._total_classes
        select_indices = np.array([0,6,7,8,9,10,12,14,17,19,22,23,25,27,31])
        # select_indices = np.array([2,4,21,24,30])
        # select_indices = np.arange(0, tot_classes)
        select_len = select_indices.shape[0]
        
        test_dataset = self.data_manager.get_dataset(select_indices, source='test', mode='test', size = 20)
        # test_dataset = self.data_manager.get_dataset(np.arange(0, tot_classes), source='test', mode='test', size = 20)
        valloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        vectors, y_true = self._extract_vectors(valloader)
        
        if showcenters:
            # fc_weight=self._network.fc.weight.data.cpu().detach().numpy()[select_indices,-self._network.out_dim:]
            proxy_fc_weight = self._network.fc.weight.data.cpu().detach().numpy()[select_indices,:]
            proxy_fc_weight = np.reshape(proxy_fc_weight, (select_len, self._cur_task+1, self._network.out_dim))
            proxy_fc = []
            for i,index in enumerate(select_indices):
                if self._cur_task != 0:
                    proxy_fc.append(np.sum(proxy_fc_weight[i], axis=0)*self._network.alpha/self._cur_task + proxy_fc_weight[i][index//self.inc]*(1-self._network.alpha/self._cur_task))
                else:
                    proxy_fc.append(proxy_fc_weight[i][0])
            proxy_fc = np.array(proxy_fc)
            print(proxy_fc.shape)
               
            vectors=np.vstack([vectors,proxy_fc])

        if Normalize:
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        embedding = umap.UMAP(min_dist=0.3,
                      metric='cosine').fit_transform(vectors)
        
        if showcenters:
            clssscenters=embedding[-select_len:,:]
            centerlabels=select_indices
            embedding=embedding[:-select_len,:]
        plt.figure(figsize=(10, 6))
        scat = []
        ncol = 1
        cmap=plt.cm.get_cmap("tab20")
        for clss in range(select_len):
            color = cmap(clss % 20)
            if clss >=20:
                scat.append(plt.scatter(embedding[clss*20:(clss+1)*20,0],embedding[clss*20:(clss+1)*20,1],marker='v',label=class_name[select_indices[clss]],s=10,color=color))
                ncol = 2
            else:
                scat.append(plt.scatter(embedding[clss*20:(clss+1)*20,0],embedding[clss*20:(clss+1)*20,1],label=class_name[select_indices[clss]],s=10,color=color))

        plt.legend(handles=scat,bbox_to_anchor=(1, 1),ncol=ncol, loc='upper left')
        if showcenters:
            plt.scatter(clssscenters[:,0],clssscenters[:,1],marker='*',s=25,c=centerlabels,cmap=plt.cm.get_cmap("tab20"),edgecolors='black')
            for i in range(len(centerlabels)):
                plt.annotate(class_name[select_indices[i]], (clssscenters[:,0][i],clssscenters[:,1][i]), textcoords = "offset points", xytext = (10,5), ha = "center")
        plt.tight_layout()
        plt.savefig("experiments/tsne/"+str(self.args['model_name'])+str(tot_classes)+'tsne_small_wou_2.png')
        plt.close()

    def tsne_proto(self, Normalize=False):
        import umap
        import matplotlib.pyplot as plt
        print('now draw tsne results of extracted features.')
        class_name = ['person', 'guitar', 'tv_stand', 'toilet', 'glass_box', 'monitor', 'bathtub', 'door', 'cup', 'sofa', 'tent', 'mantel', 'bed', 'piano', 'radio', 'chair', 'desk', 'bowl', 'flower_pot', 'laptop', 'plant', 'dresser', 'range_hood', 'stool', 'night_stand', 'car', 'lamp', 'curtain', 'bottle', 'airplane', 'wardrobe', 'keyboard', 'bookshelf', 'cone', 'vase', 'stairs', 'bench', 'sink', 'xbox', 'table']
        tot_classes=self._total_classes
        
        vectors=self._network.fc.weight.data.cpu().detach().numpy()[:tot_classes,-self._network.out_dim:]
        
        if Normalize:
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        embedding = umap.UMAP(min_dist=0.3,
                      metric='cosine').fit_transform(vectors)
        
        labels = [i for i in range(tot_classes)]
        plt.figure(figsize=(10, 6))
        cln = class_name[:tot_classes]
        plt.scatter(embedding[:,0],embedding[:,1],marker='*',s=25,c=labels,cmap=plt.cm.get_cmap("tab20"),edgecolors='black')
        for i in range(tot_classes):
            plt.annotate(cln[i], (embedding[:,0][i],embedding[:,1][i]), textcoords = "offset points", xytext = (10,5), ha = "center")

        plt.savefig(str(self.args['model_name'])+str(tot_classes)+'tsne_onlyproto.png')
        plt.close()

    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
            "fc_weight_data":self._network.fc.state_dict(),
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes, self.args["increment"])
        single, m_acc = accuracy_single(y_pred.T[0], y_true)
        ret["grouped"] = grouped
        # ret["single"] = single
        ret["MAcc"] = m_acc
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        return ret

    def eval_task(self):
        start_time = time.time()
        y_pred, y_true = self._eval_cnn(self.test_loader)
        end_time = time.time()
        print("Test Data Number is : ", len(y_pred))
        print(f"Eval took {end_time - start_time} seconds to run.")
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        return cnn_accy, nme_accy

    def incremental_train(self):
        pass

    def _train(self):
        pass
    
    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []

        with torch.no_grad():
            for _, _inputs, _targets in loader:
                _targets = _targets.numpy()
                if isinstance(self._network, nn.DataParallel):
                    _vectors = tensor2numpy(
                        self._network.module.extract_vector(_inputs.to(self._device))
                    )
                else:
                    _vectors = tensor2numpy(
                        self._network.extract_vector(_inputs.to(self._device))
                    )

                vectors.append(_vectors)
                targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)
    
    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per classes)".format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(
            self._targets_memory
        )
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt)
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar_unified(self, data_manager, m):
        logging.info(
            "Constructing exemplars for new classes...({} per classes)".format(m)
        )
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = (
                self._data_memory[mask],
                self._targets_memory[mask],
            )

            class_dset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(class_data, class_targets)
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )

            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            exemplar_dset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            exemplar_loader = DataLoader(
                exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        self._class_means = _class_means