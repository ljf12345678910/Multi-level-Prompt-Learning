import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import AnomalyCLIP_PromptLearner, CLIP_PromptLearner, dbscan, AnomalyCLIP_PromptLearners
from loss import FocalLoss, BinaryDiceLoss
from utils import normalize
from dataset import Dataset,Datasets
from logger import get_logger
from tqdm import tqdm
import numpy as np
import os
import random
from utils import get_transform
from AnomalyCLIP_lib.surgery import get_similarity_map,clip_feature_surgerys
from AnomalyCLIP_lib.adapter import Adapter, LinearLayer, Adapters
from torch.utils.data import ConcatDataset
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):

    logger = get_logger(args.save_path)

    preprocess, target_transform = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details = parameters)
    model.eval()

    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    


  ##########################################################################################
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), parameters)
    prompt_learner.to(device)

    prompt_learner_layer2 = AnomalyCLIP_PromptLearner(model.to("cpu"), parameters)
    prompt_learner_layer2.to(device)

    prompt_learner_layer3 = AnomalyCLIP_PromptLearner(model.to("cpu"), parameters)
    prompt_learner_layer3.to(device)

    prompt_learner_global = AnomalyCLIP_PromptLearner(model.to("cpu"), parameters)
    prompt_learner_global.to(device)

    prompt_learner_layer4 = CLIP_PromptLearner(model.to("cpu"), parameters, args.dataset)
    prompt_learner_layer4.to(device)

    prompt_learner_layer4_student = AnomalyCLIP_PromptLearner(model.to("cpu"), parameters)
    prompt_learner_layer4_student.to(device)


    adapter_layer1_image = Adapter(768, 4, 0, 1)
    adapter_layer1_image.to(device)

    adapter_layer2_image = Adapter(768, 4, 0, 1)
    adapter_layer2_image.to(device)

    adapter_layer3_image = Adapter(768, 4, 0, 1)
    adapter_layer3_image.to(device)

    adapter_layer4_image = Adapter(768, 4, 0, 2)
    adapter_layer4_image.to(device)

    model.to(device)
    model.visual.DAPM_replace(DPAM_layer = 24)
    ##########################################################################################
    optimizer = torch.optim.Adam(list(prompt_learner.parameters()), lr=args.learning_rate, betas=(0.5, 0.999), weight_decay=0.0001)
    optimizer_layer2 = torch.optim.Adam(list(prompt_learner_layer2.parameters()), lr=args.learning_rate, betas=(0.5, 0.999), weight_decay=0.0001)
    optimizer_layer3 = torch.optim.Adam(list(prompt_learner_layer3.parameters()), lr=args.learning_rate, betas=(0.5, 0.999), weight_decay=0.0001)
    optimizer_layer4 = torch.optim.Adam(list(prompt_learner_layer4.parameters()), lr=args.learning_rate, betas=(0.5, 0.999), weight_decay=0.0001)
    optimizer_global = torch.optim.Adam(list(prompt_learner_global.parameters()), lr=args.learning_rate, betas=(0.5, 0.999), weight_decay=0.0001)
    optimizer_student = torch.optim.Adam(list(prompt_learner_layer4_student.parameters()), lr=args.learning_rate, betas=(0.5, 0.999), weight_decay=0.0001)
    optimizer_adapter1_image = torch.optim.Adam(adapter_layer1_image.parameters(), lr=args.learning_rate2, betas=(0.5, 0.999), weight_decay=0.0001)
    optimizer_adapter2_image = torch.optim.Adam(adapter_layer2_image.parameters(), lr=args.learning_rate2, betas=(0.5, 0.999), weight_decay=0.0001)
    optimizer_adapter3_image = torch.optim.Adam(adapter_layer3_image.parameters(), lr=args.learning_rate2, betas=(0.5, 0.999), weight_decay=0.0001)
    optimizer_adapter4_image = torch.optim.Adam(adapter_layer4_image.parameters(), lr=args.learning_rate2, betas=(0.5, 0.999), weight_decay=0.0001)

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    
    
    model.eval()
    prompt_learner.train()
    prompt_learner_layer2.train()
    prompt_learner_layer3.train()
    prompt_learner_layer4.train()
    prompt_learner_global.train()
    prompt_learner_layer4_student.train()
    adapter_layer1_image.train()
    adapter_layer2_image.train()
    adapter_layer3_image.train()
    adapter_layer4_image.train()

    for epoch in tqdm(range(args.epoch)):
        model.eval()
        prompt_learner.train()
        prompt_learner_layer2.train()
        prompt_learner_layer3.train()
        prompt_learner_layer4.train()
        prompt_learner_global.train()
        prompt_learner_layer4_student.train()
        adapter_layer1_image.train()
        adapter_layer2_image.train()
        adapter_layer3_image.train()
        adapter_layer4_image.train()
 
        loss_list = []
        loss_layer2_list = []
        loss_layer3_list = []
        loss_layer4_list = []
        loss_student_list = []
        image_loss_list = []
        loss_global_list = []
        loss_linear1_list = []
        loss_linear2_list = []
        loss_linear3_list = []
        loss_linear4_list = []

        for items in tqdm(train_dataloader):
            image = items['img'].to(device)
            label =  items['anomaly']
            cls_id = items['cls_id']

            gt = items['img_mask'].squeeze().to(device)#shape:[8,518,518]
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            
            with torch.no_grad():
                #image_features:[8,768];patch_length:4;patch_features:[8,1370,768]
                image_features, patch_features, feature = model.encode_image(image, args.features_list, DPAM_layer = 24)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                feature = torch.mean(feature.permute(1, 0, 2),dim=0).unsqueeze(0)
                feature = feature/feature.norm(dim=-1,keepdim=True)
            
           ####################################   compound_length:8;compound_prompts_text:[4,768]
            prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)#prompts:[2,77,768];tokenized_prompts:[2,77]
            text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()#text_feature:[2,768]
            text_features = torch.stack(torch.chunk(text_features, dim = 0, chunks = 2), dim = 1)#[1,2,768]
            text_features = text_features/text_features.norm(dim=-1, keepdim=True)
            
            prompts_layer2, tokenized_prompts_layer2, compound_prompts_text_layer2 = prompt_learner_layer2(cls_id = None)#prompts:[2,77,768];tokenized_prompts:[2,77]
            text_features_layer2 = model.encode_text_learn(prompts_layer2, tokenized_prompts_layer2, compound_prompts_text_layer2).float()#text_feature:[2,768]
            text_features_layer2 = torch.stack(torch.chunk(text_features_layer2, dim = 0, chunks = 2), dim = 1)#[1,2,768]
            text_features_layer2 = text_features_layer2/text_features_layer2.norm(dim=-1, keepdim=True)

            prompts_layer3, tokenized_prompts_layer3, compound_prompts_text_layer3 = prompt_learner_layer3(cls_id = None)#prompts:[2,77,768];tokenized_prompts:[2,77]
            text_features_layer3 = model.encode_text_learn(prompts_layer3, tokenized_prompts_layer3, compound_prompts_text_layer3).float()#text_feature:[2,768]
            text_features_layer3 = torch.stack(torch.chunk(text_features_layer3, dim = 0, chunks = 2), dim = 1)#[1,2,768]
            text_features_layer3 = text_features_layer3/text_features_layer3.norm(dim=-1, keepdim=True)

            #CUDA_LAUNCH_BLOCKING=1
            prompts_layer4, tokenized_prompts_layer4, compound_prompts_text_layer4 = prompt_learner_layer4(cls_id = None)#prompts:[2,77,768];tokenized_prompts:[2,77]
            text_features_layer4 = model.encode_text_learn(prompts_layer4, tokenized_prompts_layer4, compound_prompts_text_layer4).float()#text_feature:[2,768]
            text_features_layer4 = torch.stack(torch.chunk(text_features_layer4, dim = 0, chunks = 2), dim = 1)#[1,2,768]
            text_features_layer4 = text_features_layer4/text_features_layer4.norm(dim=-1, keepdim=True)
            # text_features_layer4 = text_features_layer4.mean(dim=0).unsqueeze(0)
            text_features_layer4 = text_features_layer4[cls_id]

            prompts_global, tokenized_prompts_global, compound_prompts_text_global = prompt_learner_global(cls_id = None)#prompts:[2,77,768];tokenized_prompts:[2,77]
            text_features_global = model.encode_text_learn(prompts_global, tokenized_prompts_global, compound_prompts_text_global).float()#text_feature:[2,768]
            text_features_global = torch.stack(torch.chunk(text_features_global, dim = 0, chunks = 2), dim = 1)#[1,2,768]
            text_features_global = text_features_global/text_features_global.norm(dim=-1, keepdim=True)
            #学生prompt
            prompts_layer4_student, tokenized_prompts_layer4_student, compound_prompts_text_layer4_student = prompt_learner_layer4_student(cls_id=None)
            text_features_layer4_student = model.encode_text_learn(prompts_layer4_student, tokenized_prompts_layer4_student, compound_prompts_text_layer4_student).float()
            text_features_layer4_student = torch.stack(torch.chunk(text_features_layer4_student, dim=0, chunks=2), dim=1)
            text_features_layer4_student = text_features_layer4_student/text_features_layer4_student.norm(dim=-1, keepdim=True)

            text_probs = image_features.unsqueeze(1) @ text_features_global.permute(0, 2, 1)
            text_probs = text_probs[:, 0, ...]/0.07
            image_loss = F.cross_entropy(text_probs.squeeze(), label.long().cuda())
            image_loss_list.append(image_loss.item())

            patch_feature_layer4 = patch_features[3]/patch_features[3].norm(dim=-1, keepdim = True)
            similarity_global, _ = AnomalyCLIP_lib.compute_similaritys(patch_feature_layer4, text_features_global)
            similarity_map_global = AnomalyCLIP_lib.get_similarity_map(similarity_global[:, 1:, :], args.image_size).permute(0, 3 ,1 ,2)
            loss_global = 0
            loss_global += loss_focal(similarity_map_global, gt)
            loss_global += loss_dice(similarity_map_global[:, 1, :, :], gt)
            loss_global += loss_dice(similarity_map_global[:, 0, :, :], 1-gt)

            optimizer_global.zero_grad()
            # image_loss.backward(retain_graph=True)
            (image_loss+loss_global).backward()
            optimizer_global.step()
            loss_global_list.append(loss_global.item())
            # with torch.no_grad():
            #     text_features_global = text_features_global.detach()
            #########################################################################
            similarity_map_list = []

            patch_feature = patch_features[0]/patch_features[0].norm(dim=-1, keepdim = True)
            similarity_text, _ = AnomalyCLIP_lib.compute_similaritys(patch_feature,text_features)#[8,1370,2]0.9 * text_features + 0.1 * text_features_global
            similarity_map_text = AnomalyCLIP_lib.get_similarity_map(similarity_text[:, 1:, :], args.image_size).permute(0, 3, 1, 2)#[8,2,518,518]
            similarity_map_list.append(similarity_map_text)

            patch_feature_layer2 = patch_features[1]/patch_features[1].norm(dim=-1, keepdim = True)#0.9 * text_features_layer2 + 0.1 * text_features_global
            similarity_layer2_text, _ = AnomalyCLIP_lib.compute_similaritys(patch_feature_layer2,text_features_layer2)
            similarity_map_layer2_text = AnomalyCLIP_lib.get_similarity_map(similarity_layer2_text[:, 1:, :], args.image_size).permute(0, 3 ,1 ,2)
            similarity_map_list.append(similarity_map_layer2_text)

            patch_feature_layer3 = patch_features[2]/patch_features[2].norm(dim=-1, keepdim = True)#0.9 * text_features_layer3 + 0.1 * text_features_global
            similarity_layer3_text, _ = AnomalyCLIP_lib.compute_similaritys(patch_feature_layer3,text_features_layer3)
            similarity_map_layer3_text = AnomalyCLIP_lib.get_similarity_map(similarity_layer3_text[:, 1:, :], args.image_size).permute(0, 3 ,1 ,2)
            similarity_map_list.append(similarity_map_layer3_text)

            patch_feature_layer4 = patch_features[3]/patch_features[3].norm(dim=-1, keepdim = True)#0.9 * text_features_layer4 + 0.1 * text_features_global
            similarity_layer4_text, _ = AnomalyCLIP_lib.compute_similaritys(patch_feature_layer4,text_features_layer4)
            similarity_map_layer4_text = AnomalyCLIP_lib.get_similarity_map(similarity_layer4_text[:, 1:, :], args.image_size).permute(0, 3 ,1 ,2)
            anomaly_map_layer4_text = (similarity_map_layer4_text[...,1] + 1 - similarity_map_layer4_text[...,0])/2.0
            similarity_map_list.append(similarity_map_layer4_text)


            loss = 0
            #length:4,size:[8,2,518,518]
            loss += loss_focal(similarity_map_list[0], gt)
            loss += loss_dice(similarity_map_list[0][:, 1, :, :], gt)
            loss += loss_dice(similarity_map_list[0][:, 0, :, :], 1-gt)

            loss_layer2 = 0
            loss_layer2 += loss_focal(similarity_map_list[1], gt)
            loss_layer2 += loss_dice(similarity_map_list[1][:, 1, :, :], gt)
            loss_layer2 += loss_dice(similarity_map_list[1][:, 0, :, :], 1-gt)

            loss_layer3 = 0
            loss_layer3 += loss_focal(similarity_map_list[2], gt)
            loss_layer3 += loss_dice(similarity_map_list[2][:, 1, :, :], gt)
            loss_layer3 += loss_dice(similarity_map_list[2][:, 0, :, :], 1-gt)

            loss_layer4 = 0
            loss_layer4 += loss_focal(similarity_map_list[3], gt)
            loss_layer4 += loss_dice(similarity_map_list[3][:, 1, :, :], gt)
            loss_layer4 += loss_dice(similarity_map_list[3][:, 0, :, :], 1-gt)

            #backward and optimize
            optimizer.zero_grad()
            optimizer_layer2.zero_grad()
            optimizer_layer3.zero_grad()
            optimizer_layer4.zero_grad()
            
            (loss+loss_layer2+loss_layer3+loss_layer4).backward()

            optimizer.step()
            optimizer_layer2.step()
            optimizer_layer3.step()
            optimizer_layer4.step()

            loss_list.append(loss.item())
            loss_layer2_list.append(loss_layer2.item())
            loss_layer3_list.append(loss_layer3.item())
            loss_layer4_list.append(loss_layer4.item())
            
            with torch.no_grad():
                text_features = text_features.detach()
                text_features_layer2 = text_features_layer2.detach()
                text_features_layer3 = text_features_layer3.detach()
                text_features_layer4 = text_features_layer4.detach()
                anomaly_map_layer4_text = anomaly_map_layer4_text.detach()
            #student
            patch_feature_layer4 = patch_features[3]/patch_features[3].norm(dim=-1, keepdim = True)
            similarity_layer4_student, _ = AnomalyCLIP_lib.compute_similaritys(patch_feature_layer4,text_features_layer4_student)
            similarity_map_layer4_student = AnomalyCLIP_lib.get_similarity_map(similarity_layer4_student[:, 1:, :], args.image_size).permute(0, 3 ,1 ,2)
            anomaly_map_layer4_student = (similarity_map_layer4_student[...,1] + 1 - similarity_map_layer4_student[...,0])/2.0
            #计算student prompt的loss1：prompt和gt的损失
            alpha = args.distil
            loss_student = 0
            loss_student += loss_focal(similarity_map_layer4_student, gt)
            loss_student += loss_dice(similarity_map_layer4_student[:, 1, :, :], gt)
            loss_student += loss_dice(similarity_map_layer4_student[:, 0, :, :], 1-gt)
 
            #计算student prompt的loss2：异常图之间的损失
            loss_stu2 = F.mse_loss(anomaly_map_layer4_text,anomaly_map_layer4_student)
            optimizer_student.zero_grad()
            ((1-alpha) * loss_student + alpha * loss_stu2).backward()
            optimizer_student.step()
            loss_student_list.append(loss_student.item())
            with torch.no_grad():
                text_features_layer4_student = text_features_layer4_student.detach()
            #优化adapter
            similarity_map_lists = []

            patch1 = adapter_layer1_image(patch_features[0])
            patch_feature = patch1/patch1.norm(dim = -1, keepdim = True)#[8,1370,768]
            similarity_image, _ = AnomalyCLIP_lib.compute_similaritys(patch_feature, text_features)#[8,1370,2]
            similarity_map_image = AnomalyCLIP_lib.get_similarity_map(similarity_image[:, 1:, :], args.image_size).permute(0, 3, 1, 2)#[8,2,518,518]
            similarity_map_lists.append(similarity_map_image)

            patch2 = adapter_layer2_image(patch_features[1])
            patch_feature_layer2 = patch2/patch2.norm(dim=-1, keepdim = True)
            similarity_layer2_image, _ = AnomalyCLIP_lib.compute_similaritys(patch_feature_layer2, text_features_layer2)
            similarity_map_layer2_image = AnomalyCLIP_lib.get_similarity_map(similarity_layer2_image[:, 1:, :], args.image_size).permute(0, 3 ,1 ,2)
            similarity_map_lists.append(similarity_map_layer2_image)

            patch3 = adapter_layer3_image(patch_features[2])
            patch_feature_layer3 = patch3/patch3.norm(dim=-1, keepdim = True)
            similarity_layer3_image, _ = AnomalyCLIP_lib.compute_similaritys(patch_feature_layer3, text_features_layer3)
            similarity_map_layer3_image = AnomalyCLIP_lib.get_similarity_map(similarity_layer3_image[:, 1:, :], args.image_size).permute(0, 3 ,1 ,2)
            similarity_map_lists.append(similarity_map_layer3_image)

            patch4 = adapter_layer4_image(patch_features[3])
            patch_feature_layer4 = patch4/patch4.norm(dim=-1, keepdim = True)
            similarity_layer4_image, _ = AnomalyCLIP_lib.compute_similaritys(patch_feature_layer4, text_features_layer4_student)
            similarity_map_layer4_image = AnomalyCLIP_lib.get_similarity_map(similarity_layer4_image[:, 1:, :], args.image_size).permute(0, 3 ,1 ,2)
            # anomaly_map_layer4_image = (similarity_map_layer4_image[...,1] + 1 - similarity_map_layer4_image[...,0])/2.0
            similarity_map_lists.append(similarity_map_layer4_image)


            loss1 = 0
            #length:4,size:[8,2,518,518]
            loss1 += loss_focal(similarity_map_lists[0], gt)
            loss1 += loss_dice(similarity_map_lists[0][:, 1, :, :], gt)
            loss1 += loss_dice(similarity_map_lists[0][:, 0, :, :], 1-gt)

            loss_linear2 = 0
            loss_linear2 += loss_focal(similarity_map_lists[1], gt)
            loss_linear2 += loss_dice(similarity_map_lists[1][:, 1, :, :], gt)
            loss_linear2 += loss_dice(similarity_map_lists[1][:, 0, :, :], 1-gt)

            loss_linear3 = 0
            loss_linear3 += loss_focal(similarity_map_lists[2], gt)
            loss_linear3 += loss_dice(similarity_map_lists[2][:, 1, :, :], gt)
            loss_linear3 += loss_dice(similarity_map_lists[2][:, 0, :, :], 1-gt)

            loss_linear4 = 0
            loss_linear4 += loss_focal(similarity_map_lists[3], gt)
            loss_linear4 += loss_dice(similarity_map_lists[3][:, 1, :, :], gt)
            loss_linear4 += loss_dice(similarity_map_lists[3][:, 0, :, :], 1-gt)


            optimizer_adapter1_image.zero_grad()
            optimizer_adapter2_image.zero_grad()
            optimizer_adapter3_image.zero_grad()
            optimizer_adapter4_image.zero_grad()

            (loss1+loss_linear2+loss_linear3+loss_linear4).backward()

            optimizer_adapter1_image.step()
            optimizer_adapter2_image.step()
            optimizer_adapter3_image.step()
            optimizer_adapter4_image.step()

            loss_linear1_list.append(loss1.item())
            loss_linear2_list.append(loss_linear2.item())
            loss_linear3_list.append(loss_linear3.item())
            loss_linear4_list.append(loss_linear4.item())
                    
        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}, image_loss:{:.4f}'.format(epoch + 1, args.epoch, np.mean(loss_list), np.mean(image_loss_list)))
            logger.info('epoch [{}/{}], loss_layer2:{:.4f}, image_loss:{:.4f}'.format(epoch + 1, args.epoch, np.mean(loss_layer2_list), np.mean(image_loss_list)))
            logger.info('epoch [{}/{}], loss_layer3:{:.4f}, image_loss:{:.4f}'.format(epoch + 1, args.epoch, np.mean(loss_layer3_list), np.mean(image_loss_list)))
            logger.info('epoch [{}/{}], loss_layer4:{:.4f}, image_loss:{:.4f}'.format(epoch + 1, args.epoch, np.mean(loss_layer4_list), np.mean(image_loss_list)))
            logger.info('epoch [{}/{}], loss_global:{:.4f}'.format(epoch + 1, args.epoch, np.mean(loss_global_list)))
            logger.info('epoch [{}/{}], loss_student:{:.4f}'.format(epoch + 1, args.epoch, np.mean(loss_student_list)))
            logger.info('epoch [{}/{}], loss1:{:.4f}'.format(epoch + 1, args.epoch, np.mean(loss_linear1_list)))
            logger.info('epoch [{}/{}], loss_linear2:{:.4f}'.format(epoch + 1, args.epoch, np.mean(loss_linear2_list)))
            logger.info('epoch [{}/{}], loss_linear3:{:.4f}'.format(epoch + 1, args.epoch, np.mean(loss_linear3_list)))
            logger.info('epoch [{}/{}], loss_linear4:{:.4f}'.format(epoch + 1, args.epoch, np.mean(loss_linear4_list)))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'prompt_learner_epoch_' + str(epoch + 1) + '.pth')
            ckp_path_prompt_learner_layer2 = os.path.join(args.save_path, 'prompt_learner_layer2_epoch_' + str(epoch + 1) + '.pth')
            ckp_path_prompt_learner_layer3 = os.path.join(args.save_path, 'prompt_learner_layer3_epoch_' + str(epoch + 1) + '.pth')
            # ckp_path_prompt_learner_layer4 = os.path.join(args.save_path, 'prompt_learner_layer4_epoch_' + str(epoch + 1) + '.pth')
            ckp_path_prompt_learner_layer4_student = os.path.join(args.save_path, 'prompt_learner_layer4_student_epoch_' + str(epoch + 1) + '.pth')
            ckp_path_prompt_learner_global = os.path.join(args.save_path, 'prompt_learner_global_epoch_' + str(epoch + 1) + '.pth')
            adapter_layer1_image_path = os.path.join(args.save_path, 'adapter_layer1_image_epoch_' + str(epoch + 1) + '.pth')
            adapter_layer2_image_path = os.path.join(args.save_path, 'adapter_layer2_image_epoch_' + str(epoch + 1) + '.pth')
            adapter_layer3_image_path = os.path.join(args.save_path, 'adapter_layer3_image_epoch_' + str(epoch + 1) + '.pth')
            adapter_layer4_image_path = os.path.join(args.save_path, 'adapter_layer4_image_epoch_' + str(epoch + 1) + '.pth')

            torch.save({"prompt_learner": prompt_learner.state_dict()}, ckp_path)
            torch.save({"prompt_learner_layer2": prompt_learner_layer2.state_dict()}, ckp_path_prompt_learner_layer2)
            torch.save({"prompt_learner_layer3": prompt_learner_layer3.state_dict()}, ckp_path_prompt_learner_layer3)
            # torch.save({"prompt_learner_layer4": prompt_learner_layer4.state_dict()}, ckp_path_prompt_learner_layer4)
            torch.save({"prompt_learner_layer4_student": prompt_learner_layer4_student.state_dict()}, ckp_path_prompt_learner_layer4_student)
            torch.save({"prompt_learner_global": prompt_learner_global.state_dict()}, ckp_path_prompt_learner_global)
            torch.save({"adapter_layer1_image": adapter_layer1_image.state_dict()}, adapter_layer1_image_path)
            torch.save({"adapter_layer2_image": adapter_layer2_image.state_dict()}, adapter_layer2_image_path)
            torch.save({"adapter_layer3_image": adapter_layer3_image.state_dict()}, adapter_layer3_image_path)
            torch.save({"adapter_layer4_image": adapter_layer4_image.state_dict()}, adapter_layer4_image_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="./data/visa", help="train dataset path")
    parser.add_argument("--good_data_path", type=str, default="./data/visa", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoint', help='path to save results')


    parser.add_argument("--dataset", type=str, default='mvtec', help="train dataset name")
    parser.add_argument("--dataset_good", type=str, default='mvtec', help="train dataset name")

    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")

    parser.add_argument("--epoch", type=int, default=15, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--learning_rate2", type=float, default=0.0001, help="learning rate2")
    parser.add_argument("--distil", type = float, default=1.0, help="distil")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
