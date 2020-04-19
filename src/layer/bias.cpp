// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "bias.h"

void *Bias_ctor(void *_self, va_list *args)
{
    Layer *self = (Layer *)_self;

    self->one_blob_only = true;
    self->support_inplace = true;

    return _self;
}

int Bias_load_param(void *_self, const ParamDict& pd)
{
    Bias *self = (Bias *)_self;

    self->bias_data_size = pd.get(0, 0);

    return 0;
}

int Bias_load_model(void *_self, const ModelBin& mb)
{
    Bias *self = (Bias *)_self;

    self->bias_data = mb.load(self->bias_data_size, 1);
    if (self->bias_data.empty())
        return -100;

    return 0;
}

int Bias_forward_inplace(void *_self, Mat& bottom_top_blob, const Option& opt)
{
    Bias *self = (Bias *)_self;

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q=0; q<channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        float bias = self->bias_data[q];

        for (int i=0; i<size; i++)
        {
            ptr[i] += bias;
        }
    }

    return 0;
}
