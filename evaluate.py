import torch 
from torch import nn 
from torch.nn import functional as F
import numpy as np  
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import AnomalyCLIP_lib
import torch 
from torch import nn 
from torch.nn import functional as F
import numpy as np  
from sklearn.metrics import average_precision_score


def evaluate(test_dataloader, model,  text1, text2, text3, text4, linear1, linear2, linear3, linear4, device, args, obj_list):
    model.eval()
    text1.eval()
    text2.eval()
    text3.eval()
    text4.eval()
    linear1.eval()
    linear2.eval()
    linear3.eval()
    linear4.eval()
    results = {}
    results['cls_names'] = []
    results['imgs_masks'] = []
    results['anomaly_maps'] = []
    id = 0
    for items in test_dataloader:
        id = id + 1
        image = items['img'].to(device)
        cls_name = items['cls_name']
        results['cls_names'].append(cls_name[0])
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results['imgs_masks'].append(gt_mask)  # px
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features, patch_features, feature = model.encode_image(image, args.features_list, DPAM_layer = 24)
            

            prompts, tokenized_prompts, compound_prompts_text = text1(cls_id = None)#prompts:[2,77,768];tokenized_prompts:[2,77]
            text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()#text_feature:[2,768]
            text_features = torch.stack(torch.chunk(text_features, dim = 0, chunks = 2), dim = 1)#[1,2,768]
            text_features = text_features/text_features.norm(dim=-1, keepdim=True)
            
            prompts_layer2, tokenized_prompts_layer2, compound_prompts_text_layer2 = text2(cls_id = None)#prompts:[2,77,768];tokenized_prompts:[2,77]
            text_features_layer2 = model.encode_text_learn(prompts_layer2, tokenized_prompts_layer2, compound_prompts_text_layer2).float()#text_feature:[2,768]
            text_features_layer2 = torch.stack(torch.chunk(text_features_layer2, dim = 0, chunks = 2), dim = 1)#[1,2,768]
            text_features_layer2 = text_features_layer2/text_features_layer2.norm(dim=-1, keepdim=True)
            
            prompts_layer3, tokenized_prompts_layer3, compound_prompts_text_layer3 = text3(cls_id = None)#prompts:[2,77,768];tokenized_prompts:[2,77]
            text_features_layer3 = model.encode_text_learn(prompts_layer3, tokenized_prompts_layer3, compound_prompts_text_layer3).float()#text_feature:[2,768]
            text_features_layer3 = torch.stack(torch.chunk(text_features_layer3, dim = 0, chunks = 2), dim = 1)#[1,2,768]
            text_features_layer3 = text_features_layer3/text_features_layer3.norm(dim=-1, keepdim=True)
            
            prompts_layer4, tokenized_prompts_layer4, compound_prompts_text_layer4 = text4(cls_id = None)#prompts:[2,77,768];tokenized_prompts:[2,77]
            text_features_layer4 = model.encode_text_learn(prompts_layer4, tokenized_prompts_layer4, compound_prompts_text_layer4).float()#text_feature:[2,768]
            text_features_layer4 = torch.stack(torch.chunk(text_features_layer4, dim = 0, chunks = 2), dim = 1)#[1,2,768]
            text_features_layer4 = text_features_layer4/text_features_layer4.norm(dim=-1, keepdim=True)
            
            anomaly_map_list = []
            patch1 = linear1(patch_features[0])
            patch_feature = patch1/patch1.norm(dim = -1, keepdim = True)#[8,1370,768]
            #patch_feature = patch_features[0]/patch_features[0].norm(dim=-1,keepdim=True)
            similarity, _ = AnomalyCLIP_lib.compute_similaritys(patch_feature, text_features)#[8,1370,2]
            similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)#[8,2,518,518]
            anomaly_map_layer1 = (similarity_map[...,1] + 1 - similarity_map[...,0])/2.0
            anomaly_map_list.append(anomaly_map_layer1.cpu().numpy())
            
            patch2 = linear2(patch_features[1])
            patch_feature_layer2 = patch2/patch2.norm(dim=-1, keepdim = True)
            #patch_feature_layer2 = patch_features[1]/patch_features[1].norm(dim=-1,keepdim=True)
            similarity_layer2, _ = AnomalyCLIP_lib.compute_similaritys(patch_feature_layer2, text_features_layer2)
            similarity_map_layer2 = AnomalyCLIP_lib.get_similarity_map(similarity_layer2[:, 1:, :], args.image_size)
            anomaly_map_layer2 = (similarity_map_layer2[...,1] + 1 - similarity_map_layer2[...,0])/2.0
            anomaly_map_list.append(anomaly_map_layer2.cpu().numpy())

            patch3 = linear3(patch_features[2])
            patch_feature_layer3 = patch3/patch3.norm(dim=-1, keepdim = True)
            # patch_feature_layer3 = patch_features[2]/patch_features[2].norm(dim=-1,keepdim=True)
            similarity_layer3, _ = AnomalyCLIP_lib.compute_similaritys(patch_feature_layer3, text_features_layer3)
            similarity_map_layer3 = AnomalyCLIP_lib.get_similarity_map(similarity_layer3[:, 1:, :], args.image_size)
            anomaly_map_layer3 = (similarity_map_layer3[...,1] + 1 - similarity_map_layer3[...,0])/2.0
            anomaly_map_list.append(anomaly_map_layer3.cpu().numpy())

            patch4 = linear4(patch_features[3])
            patch_feature_layer4 = patch4/patch4.norm(dim=-1, keepdim = True)
            # patch_feature_layer4 = patch_features[3]/patch_features[3].norm(dim=-1,keepdim=True)
            similarity_layer4, _ = AnomalyCLIP_lib.compute_similaritys(patch_feature_layer4, text_features_layer4)
            similarity_map_layer4 = AnomalyCLIP_lib.get_similarity_map(similarity_layer4[:, 1:, :], args.image_size)
            anomaly_map_layer4 = (similarity_map_layer4[...,1] + 1 - similarity_map_layer4[...,0])/2.0
            anomaly_map_list.append(anomaly_map_layer4.cpu().numpy())
            # anomaly_map = torch.stack(anomaly_map_list)
            anomaly_map =  np.sum(anomaly_map_list, axis=0)
            results['anomaly_maps'].append(anomaly_map)
    # metrics
    ap_px_ls = []
    for obj in obj_list:
        table = []
        gt_px = []
        pr_px = []
        table.append(obj)
        for idxes in range(len(results['cls_names'])):
            if results['cls_names'][idxes] == obj:
                gt_px.append(results['imgs_masks'][idxes].squeeze(1).numpy())
                pr_px.append(results['anomaly_maps'][idxes])
        gt_px = np.array(gt_px)
        pr_px = np.array(pr_px)
        ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())
        ap_px_ls.append(ap_px)

    ap_mean = np.mean(ap_px_ls)
    model.train()
    text1.train()
    text2.train()
    text3.train()
    text4.train()
    linear1.train()
    linear2.train()
    linear3.train()
    linear4.train()
    del results, gt_px, pr_px
    return ap_mean 
