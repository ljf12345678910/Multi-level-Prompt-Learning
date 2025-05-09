import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import AnomalyCLIP_PromptLearner, CLIP_PromptLearner, dbscan, AnomalyCLIP_PromptLearners
from loss import FocalLoss, BinaryDiceLoss
from utils import normalize
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
from AnomalyCLIP_lib.surgery import clip_feature_surgery, get_similarity_map, clip_feature_surgerys
import os
import random
import numpy as np
from tabulate import tabulate
from utils import get_transform
from AnomalyCLIP_lib.adapter import Adapter, LinearLayer, Adapters
from AnomalyCLIP_lib.model_load import compute_similaritys
from visualization import visualizer,visualizer1,visualizer2,visualizer3,visualizer4
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from visualization import visualizer

from metrics import image_level_metrics, pixel_level_metrics
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
def test(args):
    img_size = args.image_size
    features_list = args.features_list
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset

    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}
    
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details = AnomalyCLIP_parameters)
    model.eval()

    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list


    results = {}
    metrics = {}
    for obj in obj_list:
        results[obj] = {}
        results[obj]['gt_sp'] = []
        results[obj]['pr_sp'] = []
        results[obj]['imgs_masks'] = []
        results[obj]['anomaly_maps'] = []
        metrics[obj] = {}
        metrics[obj]['pixel-auroc'] = 0
        metrics[obj]['pixel-aupro'] = 0
        metrics[obj]['pixel-ap'] = 0
        metrics[obj]['pixel-f1'] = 0
        metrics[obj]['image-auroc'] = 0
        metrics[obj]['image-ap'] = 0

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    checkpoint = torch.load(args.checkpoint_path)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)

    prompt_learner_layer2 = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    checkpoint_layer2 = torch.load(args.checkpoint_layer2_path)
    prompt_learner_layer2.load_state_dict(checkpoint_layer2["prompt_learner_layer2"])
    prompt_learner_layer2.to(device)

    prompt_learner_layer3 = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    checkpoint_layer3 = torch.load(args.checkpoint_layer3_path)
    prompt_learner_layer3.load_state_dict(checkpoint_layer3["prompt_learner_layer3"])
    prompt_learner_layer3.to(device)

    prompt_learner_global = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    checkpoint_global = torch.load(args.checkpoint_global_path)
    prompt_learner_global.load_state_dict(checkpoint_global["prompt_learner_global"])
    prompt_learner_global.to(device)

    # prompt_learner_layer4 = CLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters, args.dataset, "test")
    # checkpoint_layer4 = torch.load(args.checkpoint_layer4_path)
    # prompt_learner_layer4.load_state_dict(checkpoint_layer4["prompt_learner_layer4"])
    # prompt_learner_layer4.to(device)
    prompt_learner_layer4 = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    checkpoint_layer4 = torch.load(args.checkpoint_layer4_student_path)
    prompt_learner_layer4.load_state_dict(checkpoint_layer4["prompt_learner_layer4_student"])
    prompt_learner_layer4.to(device)

    adapter_layer1_image = Adapter(768, 4, 0, 1)
    checkpoint_adapter_layer1_image = torch.load(args.adapter_layer1_image_path)
    adapter_layer1_image.load_state_dict(checkpoint_adapter_layer1_image["adapter_layer1_image"])
    adapter_layer1_image.to(device)

    adapter_layer2_image = Adapter(768, 4, 0, 1)
    checkpoint_adapter_layer2_image = torch.load(args.adapter_layer2_image_path)
    adapter_layer2_image.load_state_dict(checkpoint_adapter_layer2_image["adapter_layer2_image"])
    adapter_layer2_image.to(device)

    adapter_layer3_image = Adapter(768, 4, 0, 1)
    checkpoint_adapter_layer3_image = torch.load(args.adapter_layer3_image_path)
    adapter_layer3_image.load_state_dict(checkpoint_adapter_layer3_image["adapter_layer3_image"])
    adapter_layer3_image.to(device)

    adapter_layer4_image = Adapter(768, 4, 0, 2)
    checkpoint_adapter_layer4_image = torch.load(args.adapter_layer4_image_path)
    adapter_layer4_image.load_state_dict(checkpoint_adapter_layer4_image["adapter_layer4_image"])
    adapter_layer4_image.to(device)

    model.to(device)
    model.visual.DAPM_replace(DPAM_layer = 24)
    for idx, items in enumerate(tqdm(test_dataloader)):
        image = items['img'].to(device)
        cls_name = items['cls_name']
        cls_id = items['cls_id']
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name[0]]['imgs_masks'].append(gt_mask)  # px
        results[cls_name[0]]['gt_sp'].extend(items['anomaly'].detach().cpu())

        with torch.no_grad():
            #image:[1,768],patch_len:4,patch[0]:[1,1370,768]
            image_features, patch_features, feature = model.encode_image(image, features_list, DPAM_layer = 24)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            feature = torch.mean(feature.permute(1, 0, 2),dim=0).unsqueeze(0)
            feature = feature/feature.norm(dim=-1,keepdim=True)
            
            # patch1 = patch_features[0] / patch_features[0].norm(dim=-1, keepdim=True)
            # patch1 = torch.mean(patch1,dim=0).unsqueeze(0)
            prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)#prompts:[2,77,768];tokenized_prompts:[2,77]
            text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()#text_feature:[2,768]
            text_features = torch.stack(torch.chunk(text_features, dim = 0, chunks = 2), dim = 1)#[1,2,768]
            text_features = text_features/text_features.norm(dim=-1, keepdim=True)
            # text_features = text_features[cls_id]

            # patch2 = patch_features[1] / patch_features[1].norm(dim=-1, keepdim=True)
            # patch2 = torch.mean(patch2,dim=0).unsqueeze(0)
            prompts_layer2, tokenized_prompts_layer2, compound_prompts_text_layer2 = prompt_learner_layer2(cls_id = None)#prompts:[2,77,768];tokenized_prompts:[2,77]
            text_features_layer2 = model.encode_text_learn(prompts_layer2, tokenized_prompts_layer2, compound_prompts_text_layer2).float()#text_feature:[2,768]
            text_features_layer2 = torch.stack(torch.chunk(text_features_layer2, dim = 0, chunks = 2), dim = 1)#[1,2,768]
            text_features_layer2 = text_features_layer2/text_features_layer2.norm(dim=-1, keepdim=True)
            # text_features_layer2 = text_features_layer2[cls_id]


            # patch3 = patch_features[2] / patch_features[2].norm(dim=-1, keepdim=True)
            # patch3 = torch.mean(patch3,dim=0).unsqueeze(0)
            prompts_layer3, tokenized_prompts_layer3, compound_prompts_text_layer3 = prompt_learner_layer3(cls_id = None)#prompts:[2,77,768];tokenized_prompts:[2,77]
            text_features_layer3 = model.encode_text_learn(prompts_layer3, tokenized_prompts_layer3, compound_prompts_text_layer3).float()#text_feature:[2,768]
            text_features_layer3 = torch.stack(torch.chunk(text_features_layer3, dim = 0, chunks = 2), dim = 1)#[1,2,768]
            text_features_layer3 = text_features_layer3/text_features_layer3.norm(dim=-1, keepdim=True)
            # text_features_layer3 = text_features_layer3[cls_id]

            # patch4 = patch_features[3] / patch_features[3].norm(dim=-1, keepdim=True)
            # patch4 = torch.mean(patch4,dim=0).unsqueeze(0)
            prompts_layer4, tokenized_prompts_layer4, compound_prompts_text_layer4 = prompt_learner_layer4(cls_id = None)#prompts:[2,77,768];tokenized_prompts:[2,77]
            text_features_layer4 = model.encode_text_learn(prompts_layer4, tokenized_prompts_layer4, compound_prompts_text_layer4).float()#text_feature:[2,768]
            text_features_layer4 = torch.stack(torch.chunk(text_features_layer4, dim = 0, chunks = 2), dim = 1)#[1,2,768]
            text_features_layer4 = text_features_layer4/text_features_layer4.norm(dim=-1, keepdim=True)
            # text_features_layer4 = torch.mean(text_features_layer4,dim=0).unsqueeze(0)
            # print(text_features_layer4.shape)
            # text_features_layer4 = dbscan(text_features_layer4, args.dataset, device)
            # text_features_layer4 = text_features_layer4[cls_id]

            prompts_layer_global, tokenized_prompts_global, compound_prompts_text_global = prompt_learner_global(cls_id = None)#prompts:[2,77,768];tokenized_prompts:[2,77]
            text_features_global = model.encode_text_learn(prompts_layer_global, tokenized_prompts_global, compound_prompts_text_global).float()#text_feature:[2,768]
            text_features_global = torch.stack(torch.chunk(text_features_global, dim = 0, chunks = 2), dim = 1)#[1,2,768]
            text_features_global = text_features_global/text_features_global.norm(dim=-1, keepdim=True)
    
            #text_feature = torch.cat((text_features_ad,text_features_layer2_ad,text_features_layer3_ad,text_features_layer4_ad),dim=0)
            
            text_probs = image_features @ text_features_global.permute(0, 2, 1)
            text_probs = (text_probs/0.07).softmax(-1)
            text_probs = text_probs[:, 0, 1]
            anomaly_map_list = []
            
            patch1 = adapter_layer1_image(patch_features[0])
            patch_feature = patch1/patch1.norm(dim = -1, keepdim = True)#[8,1370,768]
            #patch_feature = patch_features[0]/patch_features[0].norm(dim=-1,keepdim=True)
            similarity, _ = AnomalyCLIP_lib.compute_similaritys(patch_feature, text_features)#[8,1370,2]
            similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)#[8,2,518,518]
            anomaly_map_layer1 = (similarity_map[...,1] + 1 - similarity_map[...,0])/2.0
            anomaly_map_list.append(anomaly_map_layer1)
            
            patch2 = adapter_layer2_image(patch_features[1])
            patch_feature_layer2 = patch2/patch2.norm(dim=-1, keepdim = True)
            #patch_feature_layer2 = patch_features[1]/patch_features[1].norm(dim=-1,keepdim=True)
            similarity_layer2, _ = AnomalyCLIP_lib.compute_similaritys(patch_feature_layer2, text_features_layer2)
            similarity_map_layer2 = AnomalyCLIP_lib.get_similarity_map(similarity_layer2[:, 1:, :], args.image_size)
            anomaly_map_layer2 = (similarity_map_layer2[...,1] + 1 - similarity_map_layer2[...,0])/2.0
            anomaly_map_list.append(anomaly_map_layer2)

            patch3 = adapter_layer3_image(patch_features[2])
            patch_feature_layer3 = patch3/patch3.norm(dim=-1, keepdim = True)
            # patch_feature_layer3 = patch_features[2]/patch_features[2].norm(dim=-1,keepdim=True)
            similarity_layer3, _ = AnomalyCLIP_lib.compute_similaritys(patch_feature_layer3, text_features_layer3)
            similarity_map_layer3 = AnomalyCLIP_lib.get_similarity_map(similarity_layer3[:, 1:, :], args.image_size)
            anomaly_map_layer3 = (similarity_map_layer3[...,1] + 1 - similarity_map_layer3[...,0])/2.0
            anomaly_map_list.append(anomaly_map_layer3)

            patch4 = adapter_layer4_image(patch_features[3])
            patch_feature_layer4 = patch4/patch4.norm(dim=-1, keepdim = True)
            # patch_feature_layer4 = patch_features[3]/patch_features[3].norm(dim=-1,keepdim=True)
            similarity_layer4, _ = AnomalyCLIP_lib.compute_similaritys(patch_feature_layer4, text_features_layer4)
            similarity_map_layer4 = AnomalyCLIP_lib.get_similarity_map(similarity_layer4[:, 1:, :], args.image_size)
            anomaly_map_layer4 = (similarity_map_layer4[...,1] + 1 - similarity_map_layer4[...,0])/2.0
            anomaly_map_list.append(anomaly_map_layer4)

            text_feature = torch.cat((text_features, text_features_layer2, text_features_layer3, text_features_layer4),dim=0)
            anomaly_map = torch.stack(anomaly_map_list)
            #add feature surgery
            features = [feature / feature.norm(dim=-1, keepdim=True) for feature in patch_features]
            # text_feature = torch.cat((text_features, text_features_layer2, text_features_layer3, text_features_layer4),dim=0)
            similarity = clip_feature_surgery(features, text_feature)
            # print(similarity.shape)
            similarity_map = get_similarity_map(similarity[:, 1:, :], (args.image_size, args.image_size), False)
            # print(similarity_map.shape)
            anomaly_maps = similarity_map[:, :, :, 1]

            anomaly_map =  anomaly_map.sum(dim=0) + anomaly_maps#+ anomaly_maps

            results[cls_name[0]]['pr_sp'].extend(text_probs.detach().cpu())
            anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = args.sigma)) for i in anomaly_map.detach().cpu()], dim = 0 )
            results[cls_name[0]]['anomaly_maps'].append(anomaly_map)
            # visualizer(items['img_path'], anomaly_map.detach().cpu().numpy(), args.image_size, args.save_path, cls_name)
            # visualizer1(items['img_path'], anomaly_map.detach().cpu().numpy(), args.image_size, args.save_path, cls_name,)
            # visualizer2(items['img_path'], anomaly_map.detach().cpu().numpy(), args.image_size, args.save_path, cls_name,)
            # visualizer3(items['img_path'], anomaly_map.detach().cpu().numpy(), args.image_size, args.save_path, cls_name,)
            # visualizer4(items['img_path'], anomaly_map.detach().cpu().numpy(), args.image_size, args.save_path, cls_name,)

    table_ls = []
    image_auroc_list = []
    image_ap_list = []
    image_f1_list = []
    pixel_ap_list = []
    pixel_f1_list = []
    pixel_auroc_list = []
    pixel_aupro_list = []
    for obj in obj_list:
        table = []
        table.append(obj)
        results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks'])
        results[obj]['anomaly_maps'] = torch.cat(results[obj]['anomaly_maps']).detach().cpu().numpy()
        if args.metrics == 'image-level':
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap) 
        elif args.metrics == 'pixel-level':
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(pixel_aupro * 100, decimals=1)))
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
        elif args.metrics == 'image-pixel-level':
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            image_f1 = image_level_metrics(results, obj, 'image-f1')
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            pixel_ap = pixel_level_metrics(results, obj, "pixel-ap")
            pixel_f1 = pixel_level_metrics(results, obj, "pixel-f1")
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(pixel_aupro * 100, decimals=1)))
            table.append(str(np.round(pixel_ap * 100, decimals=1)))
            table.append(str(np.round(pixel_f1 * 100, decimals=1)))
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            table.append(str(np.round(image_f1 * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap) 
            image_f1_list.append(image_f1)
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
            pixel_ap_list.append(pixel_ap)
            pixel_f1_list.append(pixel_f1)
        table_ls.append(table)

    if args.metrics == 'image-level':
        # logger
        table_ls.append(['mean', 
                        str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
        results = tabulate(table_ls, headers=['objects', 'image_auroc', 'image_ap'], tablefmt="pipe")
    elif args.metrics == 'pixel-level':
        # logger
        table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1))
                       ])
        results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro'], tablefmt="pipe")
    elif args.metrics == 'image-pixel-level':
        # logger
        table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1)), 
                        str(np.round(np.mean(pixel_ap_list) * 100, decimals=1)),
                        str(np.round(np.mean(pixel_f1_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_ap_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_f1_list) * 100, decimals=1))])
        results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro', 'pixel_ap', 'pixel_f1',
                                              'image_auroc', 'image_ap', 'image_f1'], tablefmt="pipe")
    logger.info("\n%s", results)

# --linear_layer1_path ${save_dir}linear_layer1_epoch_15.pth \
if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default="./data/visa", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoint/', help='path to checkpoint')
    parser.add_argument("--checkpoint_layer2_path", type=str, default='./checkpoint/', help='path to checkpoint')
    parser.add_argument("--checkpoint_layer3_path", type=str, default='./checkpoint/', help='path to checkpoint')
    parser.add_argument("--checkpoint_layer4_student_path", type=str, default='./checkpoint/', help='path to checkpoint')
    parser.add_argument("--checkpoint_global_path", type=str, default='./checkpoint/', help='path to checkpoint')
    parser.add_argument("--adapter_layer1_image_path", type=str, default='./checkpoint/', help='path to checkpoint')
    parser.add_argument("--adapter_layer2_image_path", type=str, default='./checkpoint/', help='path to checkpoint')
    parser.add_argument("--adapter_layer3_image_path", type=str, default='./checkpoint/', help='path to checkpoint')
    parser.add_argument("--adapter_layer4_image_path", type=str, default='./checkpoint/', help='path to checkpoint')
    # model
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int,  nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--metrics", type=str, default='image-pixel-level')
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="zero shot")
    
    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    test(args)
