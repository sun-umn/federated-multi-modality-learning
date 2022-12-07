# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os.path

import torch
from pt_constants import PTConstants
from simple_network import BertModel
from dataset import DataSequence
from parse_metric_summary import parse_summary
from torch import nn
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
import pandas as pd
from transformers import BertTokenizerFast

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.abstract.model import (
    ModelLearnable,
    ModelLearnableKey,
    make_model_learnable,
    model_learnable_to_dxo,
)
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.pt.pt_fed_utils import PTModelPersistenceFormatManager

from seqeval.metrics import classification_report



class PTLearner(Learner):
    def __init__(self, data_path="~/data", lr=0.01, epochs=5, bs=4, exclude_vars=None, analytic_sender_id="analytic_sender"):
        """Simple PyTorch Learner that trains and validates a simple network on the CIFAR10 dataset.

        Args:
            lr (float, optional): Learning rate. Defaults to 0.01
            epochs (int, optional): Epochs. Defaults to 5
            exclude_vars (list): List of variables to exclude during model loading.
            analytic_sender_id: id of `AnalyticsSender` if configured as a client component.
                If configured, TensorBoard events will be fired. Defaults to "analytic_sender".
        """
        super().__init__()
        self.writer = None
        self.persistence_manager = None
        self.default_train_conf = None
        self.test_loader = None
        self.test_data = None
        self.n_iterations = None
        self.train_loader = None
        self.train_dataset = None
        self.optimizer = None
        self.loss = None
        self.device = None
        self.model = None
        self.data_path = data_path
        self.lr = lr
        self.bs = bs
        self.epochs = epochs
        self.exclude_vars = exclude_vars
        self.analytic_sender_id = analytic_sender_id
        self.dataprallel = False
        self.best_metric_higher_prefered = -float('inf')
        self.best_metric_lower_prefered = float('inf')
        self.global_round = 0 
        


    def initialize(self, parts: dict, fl_ctx: FLContext):
        client_name = fl_ctx.get_identity_name()
        self.client_name = fl_ctx.get_identity_name()
        
        import numpy as np
        import random
        seed = 0
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Training setup
        # self.model = BertModel(num_labels = len(unique_labels))
        self.model = BertModel()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.dataprallel:
            self.model = nn.parallel.DistributedDataParallel(self.model)
        self.model.to(self.device)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

        # Create dataset for training.
                
        df_train = pd.read_csv(os.path.join(self.data_path, client_name+"_train.csv"))
        df_val = pd.read_csv(os.path.join(self.data_path, client_name+"_val.csv"))


        self.train_dataset = DataSequence(df_train)
        self.test_dataset = DataSequence(df_val)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.bs, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.bs, shuffle=False)
        self.n_iterations = len(self.train_loader)

        # Set up the persistence manager to save PT model.
        # The default training configuration is used by persistence manager in case no initial model is found.
        self.default_train_conf = {"train": {"model": type(self.model).__name__}}
        self.persistence_manager = PTModelPersistenceFormatManager(
            data=self.model.module.state_dict() if self.dataprallel else self.model.state_dict(), default_train_conf=self.default_train_conf
        )

        # Tensorboard streaming setup
        self.writer = parts.get(self.analytic_sender_id)  # user configuration from config_fed_client.json
        if not self.writer:  # else use local TensorBoard writer only
            self.writer = SummaryWriter(fl_ctx.get_prop(FLContextKey.APP_ROOT))
        
        print("-"*50, f"initialize: {self.client_name}", "-"*50)
        print(
            f'''
            load data from {self.data_path}: {client_name+"_train.csv"} and {client_name+"_val.csv"}
            size of the data: 
            {client_name+"_train.csv"}: {df_train.shape[0]}
            {client_name+"_val.csv"}: {df_val.shape[0]}
            '''
        )
        

    def train(self, data: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        print("-"*50, f"train start: {self.client_name}", "-"*50)
        # Get model weights
        try:
            dxo = from_shareable(data)
        except:
            self.log_error(fl_ctx, "Unable to extract dxo from shareable.")
            return make_reply(ReturnCode.BAD_TASK_DATA)

        # Ensure data kind is weights.
        if not dxo.data_kind == DataKind.WEIGHTS:
            self.log_error(fl_ctx, f"data_kind expected WEIGHTS but got {dxo.data_kind} instead.")
            return make_reply(ReturnCode.BAD_TASK_DATA)

        # Convert weights to tensor. Run training
        torch_weights = {k: torch.as_tensor(v) for k, v in dxo.data.items()}
        # Set the model weights
        if self.dataprallel:
            self.model.module.load_state_dict(state_dict=torch_weights)
        else:
            self.model.load_state_dict(state_dict=torch_weights)
        self.local_train(fl_ctx, abort_signal)
        self.global_round += 1

        # Check the abort_signal after training.
        # local_train returns early if abort_signal is triggered.
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # Save the local model after training.
        self.save_local_model(fl_ctx, saved_name = PTConstants.PTLocalModelName)

        # Get the new state dict and send as weights
        new_weights = self.model.module.state_dict() if self.dataprallel else self.model.state_dict()
        new_weights = {k: v.cpu().numpy() for k, v in new_weights.items()}

        outgoing_dxo = DXO(
            data_kind=DataKind.WEIGHTS, data=new_weights, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self.n_iterations}
        )
        
        return outgoing_dxo.to_shareable()

    def local_train(self, fl_ctx, abort_signal):
        print("-"*50, f"local training: {self.client_name}", "-"*50)
        # Basic training
        for epoch in range(self.epochs):
            self.model.train()
            total_acc_train, total_loss_train, train_total = 0, 0, 0
            y_pred, y_true = [], []
            label_map = self.train_loader.dataset.ids_to_labels
            for batch in self.train_loader:
                if abort_signal.triggered:
                    return

                train_data, train_label = batch[0].to(self.device), batch[1].to(self.device)
                train_total += train_label.shape[0]
                mask = train_data['attention_mask'].squeeze(1).to(self.device)
                input_id = train_data['input_ids'].squeeze(1).to(self.device)
                # print(mask.shape, input_id.shape, train_label.shape)
                self.optimizer.zero_grad()
                loss, logits = self.model(input_id, mask, train_label)
                loss.backward()
                self.optimizer.step()
                
                for i in range(logits.shape[0]):
                    # remove padding tokens
                    logits_clean = logits[i][train_label[i] != -100]
                    label_clean = train_label[i][train_label[i] != -100].cpu().detach().numpy()
                    predictions = logits_clean.argmax(dim=1).cpu().detach().numpy()
                    y_pred.append([label_map[x] for x in predictions])
                    y_true.append([label_map[x] for x in label_clean])
                    acc = (predictions == label_clean).mean()
                    total_acc_train += acc
                    total_loss_train += loss.item()

            # Stream training, validation metrics at the end of each epoch
            metric_summary = classification_report(y_true, y_pred)
            metric_dict = parse_summary(metric_summary)
            metric_dict['macro avg']['loss'] = total_loss_train/train_total
            metric_dict['macro avg']['acc'] = 1.0* total_acc_train / train_total
            
            val_metric, val_loss, val_metric_summary = self.local_validate(abort_signal)
            val_metric_dict = parse_summary(val_metric_summary)
            val_metric_dict['macro avg']['loss'] = val_loss
            val_metric_dict['macro avg']['acc'] = val_metric
            
            global_epoches = self.global_round*self.epochs+epoch
            for metric_name in ['loss', 'acc', 'f1-score']:
                self.writer.add_scalars(f'{metric_name}', {
                    "train": metric_dict['macro avg'][metric_name],
                    "validation":  val_metric_dict['macro avg'][metric_name],
                    }, global_epoches)
                self.writer.add_text("summary/train", metric_summary, global_step=global_epoches)
                self.writer.add_text("summary/val", val_metric_summary, global_step=global_epoches)

            ## save the model if it hits the best record so far
            if self.best_metric_higher_prefered < val_metric_dict['macro avg']['f1-score']:
                self.save_local_model(fl_ctx, saved_name = PTConstants.PTLocalBestModelName)
                self.best_metric_higher_prefered = val_metric_dict['macro avg']['f1-score']
            
            ## log and print the evaluation results
            print(f"global epoches: {self.global_round} training : {epoch}/{self.epochs}: \n{metric_summary}\nF1-score: {metric_dict['macro avg']['acc']}", )
            print(f"global epoches: {self.global_round} val {epoch}/{self.epochs}: \n{val_metric_summary}\nF1-score: {val_metric_dict['macro avg']['acc']}")
            self.log_info(
                        fl_ctx, f"train:\n{metric_summary}\n==========================================================\nval:\n{val_metric_summary}"
                    )
            

    def get_model_for_validation(self, model_name: str, fl_ctx: FLContext) -> Shareable:
        print("-"*50, f"get model for validation: {self.client_name}", "-"*50)
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_job_id())
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            return None
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        self.persistence_manager = PTModelPersistenceFormatManager(
            data=torch.load(model_path), default_train_conf=self.default_train_conf
        )
        ml = self.persistence_manager.to_model_learnable(exclude_vars=self.exclude_vars)

        # Get the model parameters and create dxo from it
        dxo = model_learnable_to_dxo(ml)
        return dxo.to_shareable()

    def validate(self, data: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        print("-"*50, f"validate: {self.client_name}", "-"*50)
        model_owner = fl_ctx.get_identity_name()
        try:
            try:
                dxo = from_shareable(data)
            except:
                self.log_error(fl_ctx, "Error in extracting dxo from shareable.")
                return make_reply(ReturnCode.BAD_TASK_DATA)

            # Ensure data_kind is weights.
            if not dxo.data_kind == DataKind.WEIGHTS:
                self.log_exception(fl_ctx, f"DXO is of type {dxo.data_kind} but expected type WEIGHTS.")
                return make_reply(ReturnCode.BAD_TASK_DATA)

            if isinstance(dxo.data, ModelLearnable):
                dxo.data = dxo.data[ModelLearnableKey.WEIGHTS]

            # Extract weights and ensure they are tensor.
            model_owner = data.get_header(AppConstants.MODEL_OWNER, fl_ctx.get_identity_name())
            weights = {k: torch.as_tensor(v, device=self.device) for k, v in dxo.data.items()}

            if self.dataprallel:
                self.model.module.load_state_dict(weights)
            else:
                self.model.load_state_dict(weights)

            # Get validation accuracy
            val_accuracy, _, metric_summary = self.local_validate(abort_signal)
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)

            self.log_info(
                fl_ctx,
                f"Accuracy when validating {model_owner}'s model on"
                f" {fl_ctx.get_identity_name()}"
                f"s data: {val_accuracy}",
            )

            dxo = DXO(data_kind=DataKind.METRICS, data={"val_acc": val_accuracy})
            return dxo.to_shareable()
        except:
            self.log_exception(fl_ctx, f"Exception in validating model from {model_owner}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def local_validate(self, abort_signal):
        print("-"*50, f"local validate: {self.client_name}", "-"*50)
        self.model.eval()
        with torch.no_grad():
            total_acc_test = 0
            total_loss_test = 0
            test_total = 0
            y_pred = []
            y_true = []
            label_map = self.train_loader.dataset.ids_to_labels
            for test_data, test_label in self.test_loader:
                if abort_signal.triggered:
                    return
                test_total += test_label.shape[0]
                mask = test_data['attention_mask'].squeeze(1).to(self.device)
                input_id = test_data['input_ids'].squeeze(1).to(self.device)
                test_label = test_label.to(self.device)
                self.optimizer.zero_grad()

                loss, logits = self.model(input_id, mask, test_label)
                
                for i in range(logits.shape[0]):
                    logits_clean = logits[i][test_label[i] != -100]
                    label_clean = test_label[i][test_label[i] != -100].cpu().detach().numpy()

                    predictions = logits_clean.argmax(dim=1).cpu().detach().numpy()
                    y_pred.append([label_map[x.item()] for x in predictions])
                    y_true.append([label_map[x.item()] for x in label_clean])
                    acc = (predictions == label_clean).mean()

                    total_acc_test += acc.item()
                    total_loss_test += loss.item()
            

            metric = 1.0 * total_acc_test / float(test_total)
            loss = total_loss_test/test_total
            metric_summary = classification_report(y_true, y_pred)
        return metric, loss, metric_summary

    def save_local_model(self, fl_ctx: FLContext, saved_name: str):
        print("-"*50, f"save local model: {self.client_name}", "-"*50)
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        model_path = os.path.join(models_dir, saved_name)
        if self.dataprallel:
            ml = make_model_learnable(self.model.module.state_dict(), {})
        else:
            ml = make_model_learnable(self.model.state_dict(), {})
        self.persistence_manager.update(ml)
        torch.save(self.persistence_manager.to_persistence_dict(), model_path)
        
