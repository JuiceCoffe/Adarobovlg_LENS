import warnings
warnings.filterwarnings(action='ignore')
import torch
import torch.nn as nn
import os
import json
from . import builder
from ..utils import misc, dist_utils
import time
from ..utils.logger import *
from ..utils.metrics import Metrics
from ..models.losses import ContactGraspNetLoss, ContactGraspNetLossOrigin
from ..utils.meters import AverageMeter
from ..options.config_utils import load_config, load_config_origin

from ..utils.utils import transform_control_points_PCT, get_control_point_tensor, transform_acronym_control_points\
                        , send_dict_to_device
#* this is single-dual branch using ACRONYM dataset

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    
    # build dataset
    train_sampler, train_dataloader = builder.dataset_builder(args, config.dataset.train)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)
    
    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None
    
    if args.resume: 
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger=logger)
        
    # print model info
    print_log('Trainable_parameters:', logger=logger)
    print_log('=' * 25, logger = logger)
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)
    
    print_log('Untrainable_parameters:', logger = logger)
    print_log('=' * 25, logger = logger)
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)
    
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    
    # optimizer & scheduler
    optimizer = builder.build_optimizer(base_model, config)
    
    #Criterion
    L1Distance = ContactGraspNetLossOrigin(load_config_origin('base_model_param_cgnet_origin.yaml'))

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)
    scheduler = builder.build_scheduler(base_model, optimizer, config, last_epoch=start_epoch-1)
    
    # train and val
    # for training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch+1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()
        
        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['ContactGraspNetLoss'])
        num_iter = 0
        n_batches = len(train_dataloader)
        for idx, data in enumerate(train_dataloader):
            send_dict_to_device(data)
            data_time.update(time.time() - batch_start_time)
            pc = data['pc_cam']
            # pc = pc.cuda()
            try:
                pred = base_model(pc)
            except:
                print('Error in forward')
                print('data path: ', data['path'])
                exit()
                continue
            
            num_iter += 1
            loss, bin_ce_loss, width_loss, adds_loss = base_model.module.get_loss(pred, data, epoch)
            loss.backward()
            
            # bin_ce_loss = loss_info['bin_ce_loss']
            # width_loss = loss_info['width_loss']
            # adds_loss = loss_info['adds_loss']
            
            #forward
            if num_iter == config.step_per_update:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), getattr(config, 'grad_norm_clip', 10), norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()
                # iterate over all model parameters
                # print(base_model.named_parameters())
                # for name, param in base_model.named_parameters():
                #     print(name, ': ', param.grad)
                    
            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                losses.update(loss.item())
            else:
                losses.update(loss.item())
            
            if args.distributed:
                torch.cuda.synchronize()
            
            n_itr = epoch * n_batches + idx
            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/recon_loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/bin_ce_loss', bin_ce_loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/width_loss', width_loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/adds_loss', adds_loss.item(), n_itr)
                # train_writer.add_scalar('Loss/Batch/collision_loss', collision_loss.item(), n_itr)
                # train_writer.add_scalar('Loss/Batch/dir_loss', dir_loss.item(), n_itr)
                # train_writer.add_scalar('Loss/Batch/app_loss', app_loss.item(), n_itr)           
            
            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
            if n_itr % 10000 == 0:
               builder.save_checkpoint(base_model, optimizer, epoch, metrics=None, best_metrics=None, prefix=f'ckpt-iter-{n_itr}', args=args, logger = logger)
            
        if isinstance(scheduler, list):
            for item in scheduler:                
                item.step()
        else:
            scheduler.step()
        epoch_end_time = time.time()
        
            
        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/recon_loss', losses.avg(), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)
        
        if epoch % args.val_freq == 0:
            #Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, L1Distance, val_writer, args, config, logger=logger)
            
            if metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger=logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)
        if epoch % 500 == 0:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)
        if (config.max_epoch - epoch) < 2:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)
        
    if train_writer is not None and val_writer is not None:
        train_writer.close()
        val_writer.close()
                    

def validate(base_model, test_dataloader, epoch, L1Distance, val_writer, args, config, logger=None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode
    
    test_losses = AverageMeter(['Recon Loss'])
    bin_ce_losses = AverageMeter(['Bin CE Loss'])
    width_losses = AverageMeter(['Width Loss'])
    adds_losses = AverageMeter(['Adds Loss'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader)
    
    interval = n_samples // 2
    
    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            send_dict_to_device(data)
            pred = base_model(data['pc_cam'])
            loss, bin_ce_loss, width_loss, adds_loss = L1Distance(pred, data)
            # bin_ce_loss = loss_info['bin_ce_loss']
            # width_loss = loss_info['width_loss']
            # adds_loss = loss_info['adds_loss']
                
            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
            
            test_losses.update(loss.item())
            bin_ce_losses.update(bin_ce_loss.item())
            width_losses.update(width_loss.item())
            adds_losses.update(adds_loss.item())
            
            _metrics = Metrics.get(pred, data)[0]
            
            # _metrics = Metrics.get(predicted_cp, quality=pred_quality, grasp=grasps, mesh=mesh)        

            if args.distributed:
                # _metrics_tmp = []
                # for _metric in _metrics:
                #     if isinstance(_metric, torch.Tensor):
                #         _metrics_tmp.append(dist_utils.reduce_tensor(_metric, args))
                #     elif isinstance(_metric, dict):
                #         for k, v in _metric.items():
                #             _metrics_tmp.append(dist_utils.reduce_tensor(v, args))
                # _metrics = _metrics_tmp 
                _metrics = [dist_utils.reduce_tensor(_metric, args) for _metric in _metrics]
            else:
                _metrics = [_metric.item() for _metric in _metrics]
            
            if '0' not in category_metrics:
                category_metrics['0'] = AverageMeter(Metrics.names())
            category_metrics['0'].update(_metrics[0])
            
            if (idx+1) % interval == 0:
                print_log('Test[%d/%d] Losses = %s Metrics = %s' % 
                          (idx+1, n_samples, ['%.4f' % l for l in test_losses.val()], ['%.4f' % m for m in _metrics]), logger = logger)
            
        for _, v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[Validation] Epoch: %d   Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]), logger = logger)
        
        if args.distributed:
            torch.cuda.synchronize()
    
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Samples\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)
    
    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        print_log(msg, logger=logger)
    
    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)
    
    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/recon_loss', test_losses.avg(), epoch)
        val_writer.add_scalar('Loss/Epoch/bin_ce_loss', bin_ce_losses.avg(), epoch)
        val_writer.add_scalar('Loss/Epoch/width_loss', width_losses.avg(), epoch)
        val_writer.add_scalar('Loss/Epoch/adds_loss', adds_losses.avg(), epoch)
        # val_writer.add_scalar('Loss/Epoch/quality_loss', collision_losses.avg(), epoch)
        
        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)
    
    return Metrics(config.consider_metric, test_metrics.avg())