from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.cuda.amp import GradScaler,autocast
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from collections import defaultdict
import torch
import logging
import io

from imdb_prac.utils import count_parameters

logger = logging.getLogger(__name__)


class ModelPipeline:
    

    def __init__(self, train_iter, val_iter, model,optimizer,
                 warm_scheduler,scheduler):
        self.train_data = train_iter
        self.val_data = val_iter
        self.model = model
        self.optimizer=optimizer
        self.warm_scheduler=warm_scheduler
        self.scheduler=scheduler

    def amp_training(self,epochs,save_epoch,blob,eval_func=None,
                     max_norm=1,warm_step=5,per_ep_eval=5):
        """
        Training Model pipeline. Using Random sampling 
        to shuffle data.Also add lr_scheduler,amp_training,
        grad_clipping/accumulate to find the best Model.
        Args:
         save_model_dir(str):The dir used for save model state,config
        """
        epoch_pgb = tqdm.trange(epochs, desc='EPOCHS')
        scaler = GradScaler()
    
        
        #output realted training info
        logger.info('****Start to Training!****')
        logger.info(f'Model trainable params nums:{count_parameters(self.model)}')
        logger.info(f'Train example nums:{len(self.train_iter.dataset)}')
        logger.info(f'Batch size:{self.train_iter.batch_size}')
        logger.info(f'Epochs:{epochs}')
        logger.info(f'Current lr:{self.optimizer.lr}')
        logger.info(f'lr warm epochs:{warm_step}')
        logger.info(f'per_eval_epoch:{per_ep_eval}')
        logger.info(f'save steps:{save_epoch}')



         # Training model
        self.model.zero_grad()
        for ep in epoch_pgb:
            # build iter pipe
            iter_loss = 0
            self.model.train()
            for b in self.train_iter:

                #put batch example in gpu
                if next(self.model.parameters()).is_cuda:
                    inputs={k:v.to(self.device) for k,v in b.items() if v}

                with autocast():
                    output, loss = self.model(**inputs)

                iter_loss += loss.item()

                # avoid grad underflow
                scaler.scale(loss).backward()

                # avoid grad expolding
                scaler.unscale_(self.optimizer)
                if max_norm:
                    clip_grad_norm_(self.model.parameters(), max_norm=max_norm)

                # update weight & lr
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()  # clear grad


            #update learning rate
            if ep>warm_step:
                self.scheduler.step()
            else:
                self.warm_scheduler.step()

            logger.info(f'[{ep+1}/{epochs}] training loss:\
                   {iter_loss/len(self.train_iter)}')

            #save model to gcloud blob
            if (ep+1)%save_epoch==0:
                logger.info(f"save model to gcloud {blob.name}")
                buffer=io.BytesIO()
                torch.save(self.model,buffer)
                blob.upload_from_file(buffer)

             # evaluate model
            if (per_ep_eval > 0) and ((ep+1) % per_ep_eval == 0):
                metrics = self.evaluate_model(self.val_iter,eval_func)
                logger.info(f'Eval metrics for trained model in {ep+1}')
                for k, v in metrics.items():
                    logger.info(f'{k}:{v}')
                
    def evaluate_model(self, data_iter, eval_func):
        eval_loss = 0
        val_pgb = tqdm.tqdm(data_iter)
        total_metrics = defaultdict(int)

        self.model.eval()
        for b in val_pgb:
            with torch.no_grad():
                if next(self.model.parameters()).is_cuda:
                    inputs={k:v.to(self.device) for k,v in b.items() if v}
                with autocast():
                    outputs, loss = self.model(**inputs)

            eval_loss += loss.item()
            # eval
            batch_metrics = eval_func(outputs, inputs["label"])
            # update metrics
            for n, v in batch_metrics.items():
                total_metrics[n] += v
        # scale for epoch level
        total_metrics['eval_loss'] = eval_loss
        for k, v in total_metrics.items():
            total_metrics[k] = v/len(self.val_iter)

        return total_metrics