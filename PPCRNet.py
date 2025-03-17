import sys
from model.pvtv2 import pvt_v2_b2 as pvt
import torch
import torch.nn as nn
import torch.nn.functional as F

# 不采用ReLU的原因，就是希望插值不受影响
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MHSA(nn.Module):
    def __init__(self, n_dims, width=44, height=44, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)   # 空间像素
        # H x W
        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position # 原特征 + 学习到位置信息的原特征
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out
    
    
class PATM(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., mode='fc'):
        super().__init__()

        self.fc_h = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
        self.fc_w = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
        self.fc_c = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)

        self.tfc_h = nn.Conv2d(2 * dim, dim, (1, 7), stride=1, padding=(0, 7 // 2), bias=False)  # groups=dim
        self.tfc_w = nn.Conv2d(2 * dim, dim, (7, 1), stride=1, padding=(7 // 2, 0), bias=False)  # groups=dim
        self.reweight = Mlp(dim, dim // 4, dim * 3)
        # self.proj = nn.Conv2d(dim, dim, 1, 1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mode = mode

        if mode == 'fc':
            # 全连接
            self.theta_h_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=False), 
                                              nn.BatchNorm2d(dim), 
                                              nn.ReLU())
            self.theta_w_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=False), 
                                              nn.BatchNorm2d(dim), 
                                              nn.ReLU())
        else:
            # 深度卷积
            self.theta_h_conv = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),
                                              nn.BatchNorm2d(dim), 
                                              nn.ReLU())
            self.theta_w_conv = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),
                                              nn.BatchNorm2d(dim), 
                                              nn.ReLU())

    def forward(self, x):
        B, C, H, W = x.shape
        
        theta_h = self.theta_h_conv(x)
        theta_w = self.theta_w_conv(x)

        x_h = self.fc_h(x)
        x_w = self.fc_w(x)
        x_h = torch.cat([x_h * torch.cos(theta_h), x_h * torch.sin(theta_h)], dim=1)
        x_w = torch.cat([x_w * torch.cos(theta_w), x_w * torch.sin(theta_w)], dim=1)

        h = self.tfc_h(x_h)
        w = self.tfc_w(x_w)
        c = self.fc_c(x)
        a = F.adaptive_avg_pool2d(h + w + c, output_size=1)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        x = h * a[0] + w * a[1] + c * a[2]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class CFM(nn.Module):
    def __init__(self, channel):
        super(CFM, self).__init__()
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)
        self.conv6 = nn.Conv2d(3*channel, channel, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        up2_x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        x2_1 = self.conv_upsample1(up2_x1) * x2
        
        up4_x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=True)
        up2_x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
        x3_1 = self.conv_upsample2(up4_x1) * self.conv_upsample3(up2_x2) * x3
        
        up2_x1_1 = F.interpolate(x1_1, scale_factor=2, mode='bilinear', align_corners=True)
        x2_2 = torch.cat((x2_1, self.conv_upsample4(up2_x1_1)), 1)
        x2_2 = self.conv_concat2(x2_2)
        up2_x2_2 = F.interpolate(x2_2, scale_factor=2, mode='bilinear', align_corners=True)
        x3_2 = torch.cat((x3_1, self.conv_upsample5(up2_x2_2)), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x_single = self.conv5(x)
        x_mul = self.conv6(x)

        return x_single, x_mul
       
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)
    
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=15, stride=1, padding=7)  ####kernel_size=7 / 15
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out
    
class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
    
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)
        
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU()
        ))
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(CBAM(in_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(4 * out_channels + in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.GELU()
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h
    
class SAM(nn.Module):
    def __init__(self, num_in=64, plane_mid=16, mids=4, normalize=False):
        super(SAM, self).__init__()

        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = (mids) * (mids)
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

    def forward(self, x, edge):
        edge = F.interpolate(edge, (x.size()[-2], x.size()[-1]))

        n, c, h, w = x.size()
        edge = torch.nn.functional.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1)

        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge

        x_anchor1 = self.priors(x_mask)
        x_anchor2 = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)
        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)

        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
        x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=1)

        x_rproj_reshaped = x_proj_reshaped

        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
        x_n_rel = self.gcn(x_n_state)

        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
        out = x + (self.conv_extend(x_state))

        return out
    
class CascadedGroupAttention(nn.Module):
    def __init__(self, dim, key_dim=16, num_heads=4, attn_ratio=1, kernels=[5, 5, 5, 5], height=0, width=0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio

        qkvs = []
        dws = []
        for i in range(num_heads):
            qkvs.append(nn.Conv2d(dim // (num_heads), self.key_dim * 2 + self.d, kernel_size=1))
            dws.append(nn.Conv2d(self.key_dim, self.key_dim, kernels[i], 1, kernels[i]//2, groups=self.key_dim))
            # self.rel_h = nn.Parameter(torch.randn([1, num_heads, num_heads, 1, height]), requires_grad=True)
            # self.rel_w = nn.Parameter(torch.randn([1, num_heads, num_heads, width, 1]), requires_grad=True)
        self.qkvs = torch.nn.ModuleList(qkvs)
        self.dws = torch.nn.ModuleList(dws)
        self.proj = torch.nn.Sequential(nn.PReLU(), nn.Conv2d(64, 64, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        feats_in = x.chunk(len(self.qkvs), dim=1)
        feats_out = []
        feat = feats_in[0]
        for i, qkv in enumerate(self.qkvs):
            if i > 0:
                feat = feat + feats_in[i]
            feat = qkv(feat)   # 将通道扩大3倍，形成q，k，v
            q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1) # B, C/h, H, W
            q = self.dws[i](q)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)   # B, C, H x W
            attn = (
                (q.transpose(-2, -1) @ k) * 1    # scale等于小数
            )
            attn = attn.softmax(dim=-1)
            feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W) # B, C, H, W
            feats_out.append(feat)
        x = torch.cat(feats_out, 1)
        return x
    
class PPCRNet(nn.Module):
    def __init__(self):
        super(PPCRNet, self).__init__()
        self.backbone = pvt()  # [64, 128, 320, 512]
        path = '/root/autodl-tmp/polyp_project/model/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.aspp2 = ASPP(128, 64, [3, 5, 7])
        self.aspp3 = ASPP(320, 64, [3, 5, 7])
        self.aspp4 = ASPP(512, 64, [3, 5, 7])

        self.patm2_block = PATM(64, qkv_bias=False, qk_scale=None, attn_drop=0., mode='fc')
        self.patm3_block = PATM(64, qkv_bias=False, qk_scale=None, attn_drop=0., mode='fc')
        self.patm4_block = PATM(64, qkv_bias=False, qk_scale=None, attn_drop=0., mode='fc')
        
        self.translayer1 = BasicConv2d(64, 64, kernel_size=1)
        self.translayer2 = BasicConv2d(64, 64, kernel_size=1)
        self.translayer3 = BasicConv2d(64, 64, kernel_size=1)
        self.translayer4 = BasicConv2d(64, 64, kernel_size=1)
        self.bc1 = BasicConv2d(64, 64, kernel_size=1)
        self.bc2 = BasicConv2d(64, 64, kernel_size=1)
        self.cfm = CFM(64)

        self.SAM2 = SAM()
        self.SAM3 = SAM()
        self.SAM4 = SAM()

        self.msha2_44 = CascadedGroupAttention(64, 16, num_heads=4, height=44, width=44)
        self.msha3_44 = CascadedGroupAttention(64, 16, num_heads=4, height=44, width=44)
        self.msha4_44 = CascadedGroupAttention(64, 16, num_heads=4, height=44, width=44)

        self.msha2_32 = CascadedGroupAttention(64, 16, num_heads=4, height=32, width=32)
        self.msha3_32 = CascadedGroupAttention(64, 16, num_heads=4, height=32, width=32)
        self.msha4_32 = CascadedGroupAttention(64, 16, num_heads=4, height=32, width=32)

        self.msha2_56 = CascadedGroupAttention(64, 16, num_heads=4, height=56, width=56)
        self.msha3_56 = CascadedGroupAttention(64, 16, num_heads=4, height=56, width=56)
        self.msha4_56 = CascadedGroupAttention(64, 16, num_heads=4, height=56, width=56)

        # ---- reverse attention branch 4 ----
        self.ra4_conv1 = BasicConv2d(64, 32, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        self.ra4_conv3 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        self.ra4_conv4 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        self.ra4_conv5 = BasicConv2d(32, 1, kernel_size=1)

        # ---- reverse attention branch 3 ----
        self.ra3_conv1 = BasicConv2d(64, 32, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(32, 1, kernel_size=1)

        # ---- reverse attention branch 2 ----
        self.ra2_conv1 = BasicConv2d(64, 32, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(32, 1, kernel_size=1)
        
        

    def forward(self, x, rate=1):
        # backbone
        pvt = self.backbone(x)
        # original-image: 3 x 352 352
        x1 = pvt[0]  # x1: 64 x 88 x 88
        x2 = pvt[1]  # x2: 128 x 44 x 44
        x3 = pvt[2]  # x3: 320 x 22 x 22
        x4 = pvt[3]  # x4: 512 x 11 x 11

        # ---- ASPP-CBAM ----
        ap2 = self.aspp2(x2)   # torch.Size([4, 64, 44, 44])
        ap3 = self.aspp3(x3)   # torch.Size([4, 64, 22, 22])
        ap4 = self.aspp4(x4)   # torch.Size([4, 64, 11, 11])

        # ---- PAM ----
        patm2 = self.patm2_block(ap2)    # torch.Size([4, 64, 44, 44])
        patm3 = self.patm3_block(ap3)    # torch.Size([4, 64, 22, 22])
        patm4 = self.patm4_block(ap4)    # torch.Size([4, 64, 11, 11])
        
        # ---- CFM ----
        fuse1, fuse2 = self.cfm(patm4, patm3, patm2)       # torch.Size([4, 1, 44, 44]) torch.Size([4, 64, 44, 44])

        # ---- SAM ----
        sam_feature_4 = self.SAM4(fuse2, patm4)            # torch.Size([4, 64, 44, 44])

        lateral_map_5 = F.interpolate(fuse1, scale_factor=8, mode='bilinear')

        # ---- MSHA 现在x是每次更新的一个新的权重----
        if rate == 0.75:
            rv_msha_att4 = self.msha4_32(sam_feature_4) + sam_feature_4
        elif rate == 1:
            rv_msha_att4 = self.msha4_44(sam_feature_4) + sam_feature_4
        else:
            rv_msha_att4 = self.msha4_56(sam_feature_4) + sam_feature_4
        x = -1 * (torch.sigmoid(fuse1)) + 1
        x = x.expand(-1, 64, -1, -1).mul(rv_msha_att4)
        x = self.ra4_conv1(x)
        x = F.relu(self.ra4_conv2(x))
        x = F.relu(self.ra4_conv3(x))
        x = F.relu(self.ra4_conv4(x))
        ra4_feat = self.ra4_conv5(x)
        x = ra4_feat + fuse1                                     #  torch.Size([4, 1, 44, 44])
        lateral_map_4 = F.interpolate(x, scale_factor=8, mode='bilinear')   
        
        
        sam_feature_3 = self.SAM3(self.bc1(sam_feature_4), patm3)  # torch.Size([4, 64, 44, 44])
        if rate == 0.75:
            rv_msha_att3 = self.msha3_32(sam_feature_3) + sam_feature_3
        elif rate == 1:
            rv_msha_att3 = self.msha3_44(sam_feature_3) + sam_feature_3
        else: 
            rv_msha_att3 = self.msha3_56(sam_feature_3) + sam_feature_3
        crop_3 = x                             # 复制上一份注意力图
        x = -1 * (torch.sigmoid(crop_3)) + 1
        x = x.expand(-1, 64, -1, -1).mul(rv_msha_att3)
        x = self.ra3_conv1(x)
        x = F.relu(self.ra3_conv2(x))
        x = F.relu(self.ra3_conv3(x))
        ra3_feat = self.ra3_conv4(x)
        x = ra3_feat + crop_3
        lateral_map_3 = F.interpolate(x, scale_factor=8, mode='bilinear')  
        
        sam_feature_2 = self.SAM2(self.bc2(sam_feature_3), patm2)  # torch.Size([4, 64, 44, 44])
        if rate == 0.75:
            rv_msha_att2 = self.msha2_32(sam_feature_2) + sam_feature_2
        elif rate == 1:
            rv_msha_att2 = self.msha2_44(sam_feature_2) + sam_feature_2
        else:
            rv_msha_att2 = self.msha2_56(sam_feature_2) + sam_feature_2
        crop_2 = x  # 复制上一份注意力图
        x = -1 * (torch.sigmoid(crop_2)) + 1
        x = x.expand(-1, 64, -1, -1).mul(rv_msha_att2)
        x = self.ra2_conv1(x)
        x = F.relu(self.ra2_conv2(x))
        x = F.relu(self.ra2_conv3(x))
        ra2_feat = self.ra2_conv4(x)
        x = ra2_feat + crop_2
        lateral_map_2 = F.interpolate(x, scale_factor=8, mode='bilinear')  

        return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2
