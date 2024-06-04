import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyUNet(nn.Module):
    def __init__(self, in_channels, out_channel, base_channels=32):
        super().__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=13, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        ) # (C, 224, 224)

        self.down_conv1 = self.make_down_conv(base_channels*2) # (2C, 112, 112)
        self.down_conv2 = self.make_down_conv(base_channels*4) # (4C, 56, 56)
        self.down_conv3 = self.make_down_conv(base_channels*8) # (8C, 28, 28)
        self.down_conv4 = self.make_down_conv(base_channels*16) # (16C, 14, 14)
        self.down_conv5 = self.make_down_conv(base_channels*32) # (32C, 7, 7)

        self.up_conv1 = self.make_up_conv(base_channels*16) # (16C, 14, 14)
        self.up_conv2 = self.make_up_conv(base_channels*8) # (16C, 28, 28)
        self.up_conv3 = self.make_up_conv(base_channels*4) # (16C, 56, 56)
        self.up_conv4 = self.make_up_conv(base_channels*2) # (16C, 112, 112)
        self.up_conv5 = self.make_up_conv(base_channels) # (16C, 224, 224)

        
        self.out_conv = nn.Conv2d(base_channels, out_channel, kernel_size=1)
        
    def make_down_conv(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels//2, channels, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def make_up_conv(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels*3, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def up_conv(self, name, x, skip_x):
        x = torch.cat([F.interpolate(x, size=skip_x.size(-1), mode='bilinear'), skip_x], dim=1)
        x = getattr(self, f'up_conv{name}')(x)
        return x

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down_conv1(x0)
        x2 = self.down_conv2(x1)
        x3 = self.down_conv3(x2)
        x4 = self.down_conv4(x3)
        x5 = self.down_conv5(x4)

        x6 = self.up_conv(1, x5, x4)
        x7 = self.up_conv(2, x6, x3)
        x8 = self.up_conv(3, x7, x2)
        x9 = self.up_conv(4, x8, x1)
        x10 = self.up_conv(5, x9, x0)
        x11 = self.out_conv(x10)
        return x11[..., 12:-12, 12:-12]

class KalmanFuser(nn.Module):
    def __init__(self, Y, feat2d_dim=128, base_channels=32):
        super().__init__()

        self.base_channels = base_channels
        self.feat2d_dim = feat2d_dim
        # self.log_epsilon = -10
        self.Y = 80
        self.radar_feature_extractor = TinyUNet(Y, out_channel=Y, base_channels=base_channels)
        # self.camera_feature_extractor = nn.Conv2d(self.feat2d_dim*Y, base_channels, 1)
        self.register_buffer('camera_feature_extractor', torch.randn(self.Y, self.Y, 1, 1) * 0.05)
        # self.feat_to_init_state = nn.Conv2d(self.Y, self.Y*2, 1)
        # self.init_mats = nn.Parameter(torch.randn(1, self.Y*2, 1, 1, requires_grad=True))
        self.feat_to_mats = nn.Conv2d(self.Y, self.Y*2 + self.Y*4, 1)
        self.feat_to_mats_camera = nn.Conv2d(self.Y, self.Y*2, 1)
        self.z_to_radar = nn.Conv2d(self.Y, self.Y, 1)

        # for conv in [self.feat_to_mats, self.feat_to_mats_camera]:
        #     conv.weight.data.normal_(std=1e-6)

    def forward(self, feat_bev_, feat_pts):
        z_posteriors = []
        z_priors = []
        radar_mses = []

        # for step in range(5):
        #     mask = (metarad_occ_mem0[:,-1:] == (5 - step))

        #     radar_raw = (metarad_occ_mem0[:,:-1] * mask.float()).permute(0,1,3,2,4)
        #     radar_raw = radar_raw.reshape(radar_raw.shape[0], -1, *radar_raw.shape[-2:])
        #     radar_feat = self.radar_feature_extractor(radar_raw)

        #     mask = (torch.ones(1, device=metarad_occ_mem0.device, dtype=torch.bool).expand_as(metarad_occ_mem0[:,:-1]) & mask).permute(0,1,3,2,4)
        #     mask = mask.reshape(mask.shape[0], -1, *mask.shape[-2:])

        #     if step == 4:
        #         camera = feat_bev_
        #         camera_feat = self.camera_feature_extractor(camera)

        #     if step == 0:
        #         s_mean, s_logstd = self.feat_to_init_state(radar_feat).split([self.base_channels] * 2, dim=1)
        #         logH_curr, logR_curr = self.init_mats.split([self.base_channels] * 2, dim=1)

        #         s_init = (s_mean, s_logstd)
        #         pred_mu = torch.normal(s_mean, s_logstd.exp()) if self.training else s_mean
        #         pred_logsigma = self.log_epsilon
        #         res_logstd = logR_curr
        #     else:
        #         pred_mu = logF_curr.exp() * mu
        #         pred_logsigma = torch.logaddexp(2*logF_curr + logsigma, logQ_curr)
        #         res_logstd = torch.logaddexp(2*logH_curr + pred_logsigma, logR_curr)
        #     z_pred = logH_curr.exp() * pred_mu
        #     z_priors.append((z_pred, res_logstd))

        #     z_mean, z_logstd, logF_next, logH_next, logQ_next, logR_next = self.feat_to_mats(radar_feat).split([self.base_channels] * 6, dim=1)
        #     logF_next = torch.log1p(logF_next.exp())
        #     z_posteriors.append((z_mean, z_logstd))

        #     z_curr = torch.normal(z_mean, z_logstd.exp()) if self.training else z_mean
        #     radar_mses.append((self.z_to_radar(z_curr) - radar_raw)[mask].square().mean())

        #     residual = z_curr - z_pred
        #     logkalman_gain = pred_logsigma + logH_curr - res_logstd
        #     mu = pred_mu + logkalman_gain.exp() * residual
        #     logsigma = F.softplus(1 - (logkalman_gain + logH_curr).exp(), beta=10).log() + pred_logsigma

        #     if step < 4:
        #         logF_curr, logH_curr, logQ_curr, logR_curr = logF_next, logH_next, logQ_next, logR_next
        #     else:
        #         logH_cam_curr, logR_cam_curr = self.feat_to_mats_camera(radar_feat).split([self.base_channels] * 2, dim=1)

        #         camera_pred = logH_cam_curr.exp() * mu
        #         residual = camera_feat - camera_pred
        #         res_logstd = torch.logaddexp(2*logH_cam_curr + logsigma, logR_cam_curr)
        #         camera_nll = residual.square()/(2*(2*res_logstd).exp())

        #         logkalman_gain = logsigma + logH_cam_curr - res_logstd
        #         mu = mu + logkalman_gain.exp() * residual
        #         logsigma = F.softplus(1 - (logkalman_gain + logH_cam_curr).exp(), beta=10).log() + logsigma

        for step in range(4):
            # with torch.no_grad():
            #     mask = (feat_pts[:,-1:] == (5 - step))
            #     radar_raw = (feat_pts[:,:-1] * mask.float()).permute(0,1,3,2,4)
            #     radar_raw = radar_raw.reshape(radar_raw.shape[0], -1, *radar_raw.shape[-2:])
            #     mask = (torch.ones(1, device=feat_ps.device, dtype=torch.bool).expand_as(metarad_occ_mem0[:,:-1]) & mask).permute(0,1,3,2,4)
            #     mask = mask.reshape(mask.shape[0], -1, *mask.shape[-2:])
            #     # mask = (metarad_occ_mem0[:,-1:] == (5 - step)).float()
            #     # radar_raw = (metarad_occ_mem0[:,:-1] * mask).permute(0,1,3,2,4)
            #     # radar_raw = radar_raw.reshape(radar_raw.shape[0], -1, *radar_raw.shape[-2:])
            
            radar_raw = feat_pts[:, step, :, :, :]
            radar_feat = self.radar_feature_extractor(radar_raw)
            print("shape of radar_feat : ", radar_feat.shape)
            if step == 3:
                camera = feat_bev_[:,0]
                # camera_feat = self.camera_feature_extractor(camera)
                camera_feat = F.conv2d(camera, self.camera_feature_extractor)

            if step == 0:
                s_init = (None, None)
                _, _, logF_curr, logH_curr, logQ_curr, logR_curr = self.feat_to_mats(torch.zeros_like(radar_feat[..., :1, :1])).split([self.Y] * 6, dim=1)
                logF_curr = torch.log1p(logF_curr.exp())
                mu = 0
                logvar = 0
                
            pred_mu = logF_curr.exp() * mu
            pred_logvar = torch.logaddexp(2*logF_curr + logvar, logQ_curr)
            res_logvar = torch.logaddexp(2*logH_curr + pred_logvar, logR_curr)
            z_pred = logH_curr.exp() * pred_mu
            z_priors.append((z_pred, res_logvar))

            z_mean, z_logvar, logF_next, logH_next, logQ_next, logR_next = self.feat_to_mats(radar_feat).split([self.Y] * 6, dim=1)
            logF_next = torch.log1p(logF_next.exp())
            z_posteriors.append((z_mean, z_logvar))

            z_curr = torch.normal(z_mean, (0.5*z_logvar).exp()) if self.training else z_mean
            radar_mses.append(
                (self.z_to_radar(z_curr) - radar_raw).square().mean()
            )
            # radar_mses.append(
            #     ((self.z_to_radar(z_curr) - radar_raw).view(radar_raw.shape[0], -1, self.Y, *radar_raw.shape[-2:]).permute(0,1,3,2,4) * mask).square().sum() / mask.sum()
            # )

            residual = z_curr - z_pred
            logkalman_gain = pred_logvar + logH_curr - res_logvar
            mu = pred_mu + logkalman_gain.exp() * residual
            logvar = F.softplus(1 - (logkalman_gain + logH_curr).exp(), beta=10).log() + pred_logvar
            # logvar = torch.log1p(-(logkalman_gain + logH_curr).exp()) + pred_logvar

            if step < 3:
                logF_curr, logH_curr, logQ_curr, logR_curr = logF_next, logH_next, logQ_next, logR_next
            else:
                logH_cam_curr, logR_cam_curr = self.feat_to_mats_camera(radar_feat).split([self.Y] * 2, dim=1)

                camera_pred = logH_cam_curr.exp() * mu
                residual = camera_feat - camera_pred
                res_logvar = torch.logaddexp(2*logH_cam_curr + logvar, logR_cam_curr)
                camera_nll = residual.square()/(2*res_logvar.exp())

                logkalman_gain = logvar + logH_cam_curr - res_logvar
                mu = mu + logkalman_gain.exp() * residual
                logvar = F.softplus(1 - (logkalman_gain + logH_cam_curr).exp(), beta=10).log() + logvar
                # logvar = torch.log1p(-(logkalman_gain + logH_cam_curr).exp()) + logvar

        sample = torch.normal(mu, (0.5*logvar).exp()) if self.training else mu

        return sample, z_posteriors, z_priors, radar_mses, camera_nll, s_init
