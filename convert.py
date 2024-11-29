import flax
import jax
import torch
def load_params(check_point):
    state_dict = torch.load(check_point,map_location="cpu")
    state_dict = state_dict["weight"]
    params = {

    }
    params[f"emb_g.embedding"]=state_dict[f"emb_g.weight"]
    params[f"enc_p.emb_phone.kernel"]=state_dict[f"enc_p.emb_phone.weight"].transpose(0,1)
    params[f"enc_p.emb_phone.bias"]=state_dict[f"enc_p.emb_phone.bias"]
    params[f"enc_p.emb_pitch.embedding"]=state_dict[f"enc_p.emb_pitch.weight"]
    params[f"enc_p.proj.kernel"]=state_dict[f"enc_p.proj.weight"].transpose(0,2)
    params[f"enc_p.proj.bias"]=state_dict[f"enc_p.proj.bias"]
    for i in range(6):
        params[f"enc_p.encoder.attn_layers_{i}.emb_rel_k"]=state_dict[f"enc_p.encoder.attn_layers.{i}.emb_rel_k"]
        params[f"enc_p.encoder.attn_layers_{i}.emb_rel_v"]=state_dict[f"enc_p.encoder.attn_layers.{i}.emb_rel_v"]
        params[f"enc_p.encoder.attn_layers_{i}.conv_q.kernel"]=state_dict[f"enc_p.encoder.attn_layers.{i}.conv_q.weight"].transpose(0,2)
        params[f"enc_p.encoder.attn_layers_{i}.conv_q.bias"]=state_dict[f"enc_p.encoder.attn_layers.{i}.conv_q.bias"]
        params[f"enc_p.encoder.attn_layers_{i}.conv_k.kernel"]=state_dict[f"enc_p.encoder.attn_layers.{i}.conv_k.weight"].transpose(0,2)
        params[f"enc_p.encoder.attn_layers_{i}.conv_k.bias"]=state_dict[f"enc_p.encoder.attn_layers.{i}.conv_k.bias"]
        params[f"enc_p.encoder.attn_layers_{i}.conv_v.kernel"]=state_dict[f"enc_p.encoder.attn_layers.{i}.conv_v.weight"].transpose(0,2)
        params[f"enc_p.encoder.attn_layers_{i}.conv_v.bias"]=state_dict[f"enc_p.encoder.attn_layers.{i}.conv_v.bias"]
        params[f"enc_p.encoder.attn_layers_{i}.conv_o.kernel"]=state_dict[f"enc_p.encoder.attn_layers.{i}.conv_o.weight"].transpose(0,2)
        params[f"enc_p.encoder.attn_layers_{i}.conv_o.bias"]=state_dict[f"enc_p.encoder.attn_layers.{i}.conv_o.bias"]
        params[f"enc_p.encoder.norm_layers_1_{i}.scale"]=state_dict[f"enc_p.encoder.norm_layers_1.{i}.gamma"]
        params[f"enc_p.encoder.norm_layers_1_{i}.bias"]=state_dict[f"enc_p.encoder.norm_layers_1.{i}.beta"]
        params[f"enc_p.encoder.norm_layers_2_{i}.scale"]=state_dict[f"enc_p.encoder.norm_layers_2.{i}.gamma"]
        params[f"enc_p.encoder.norm_layers_2_{i}.bias"]=state_dict[f"enc_p.encoder.norm_layers_2.{i}.beta"]
        params[f"enc_p.encoder.ffn_layers_{i}.conv1.kernel"]=state_dict[f"enc_p.encoder.ffn_layers.{i}.conv_1.weight"].transpose(0,2)
        params[f"enc_p.encoder.ffn_layers_{i}.conv1.bias"]=state_dict[f"enc_p.encoder.ffn_layers.{i}.conv_1.bias"]
        params[f"enc_p.encoder.ffn_layers_{i}.conv2.kernel"]=state_dict[f"enc_p.encoder.ffn_layers.{i}.conv_2.weight"].transpose(0,2)
        params[f"enc_p.encoder.ffn_layers_{i}.conv2.bias"]=state_dict[f"enc_p.encoder.ffn_layers.{i}.conv_2.bias"]
    params[f"dec.m_source.l_linear.kernel"]=state_dict[f"dec.m_source.l_linear.weight"].transpose(0,1)
    params[f"dec.m_source.l_linear.bias"]=state_dict[f"dec.m_source.l_linear.bias"]
    params[f"dec.conv_pre.kernel"]=state_dict[f"dec.conv_pre.weight"].transpose(0,2)
    params[f"dec.conv_pre.bias"]=state_dict[f"dec.conv_pre.bias"]
    params[f"dec.conv_post.kernel"]=state_dict[f"dec.conv_post.weight"].transpose(0,2)
    params[f"dec.cond.kernel"]=state_dict[f"dec.cond.weight"].transpose(0,2)
    params[f"dec.cond.bias"]=state_dict[f"dec.cond.bias"]
    for i in range(4):
        params[f"dec.noise_convs_{i}.kernel"]=state_dict[f"dec.noise_convs.{i}.weight"].transpose(0,2)
        params[f"dec.noise_convs_{i}.bias"]=state_dict[f"dec.noise_convs.{i}.bias"]
        params[f"dec.ups_{i}.layer_instance/kernel/scale"]=state_dict[f"dec.ups.{i}.weight_g"].squeeze()
        params[f"dec.ups_{i}.layer_instance.kernel"]=state_dict[f"dec.ups.{i}.weight_v"].transpose(0,2)
        params[f"dec.ups_{i}.layer_instance.bias"]=state_dict[f"dec.ups.{i}.bias"]
    for i in range(12):
        for j in range(3):
            params[f"dec.resblocks_{i}.convs1_{j}.layer_instance/kernel/scale"]=state_dict[f"dec.resblocks.{i}.convs1.{j}.weight_g"].squeeze()
            params[f"dec.resblocks_{i}.convs1_{j}.layer_instance.kernel"]=state_dict[f"dec.resblocks.{i}.convs1.{j}.weight_v"].transpose(0,2)
            params[f"dec.resblocks_{i}.convs1_{j}.layer_instance.bias"]=state_dict[f"dec.resblocks.{i}.convs1.{j}.bias"]
            params[f"dec.resblocks_{i}.convs2_{j}.layer_instance/kernel/scale"]=state_dict[f"dec.resblocks.{i}.convs2.{j}.weight_g"].squeeze()
            params[f"dec.resblocks_{i}.convs2_{j}.layer_instance.kernel"]=state_dict[f"dec.resblocks.{i}.convs2.{j}.weight_v"].transpose(0,2)
            params[f"dec.resblocks_{i}.convs2_{j}.layer_instance.bias"]=state_dict[f"dec.resblocks.{i}.convs2.{j}.bias"]
    for i in range(0,8,2):
        params[f"flow.flows_{i}.pre.kernel"]=state_dict[f"flow.flows.{i}.pre.weight"].transpose(0,2)
        params[f"flow.flows_{i}.pre.bias"]=state_dict[f"flow.flows.{i}.pre.bias"]
        params[f"flow.flows_{i}.post.kernel"]=state_dict[f"flow.flows.{i}.post.weight"].transpose(0,2)
        params[f"flow.flows_{i}.post.bias"]=state_dict[f"flow.flows.{i}.post.bias"]
        params[f"flow.flows_{i}.enc.cond_layer.layer_instance/kernel/scale"]=state_dict[f"flow.flows.{i}.enc.cond_layer.weight_g"].squeeze()
        params[f"flow.flows_{i}.enc.cond_layer.layer_instance.kernel"]=state_dict[f"flow.flows.{i}.enc.cond_layer.weight_v"].transpose(0,2)
        params[f"flow.flows_{i}.enc.cond_layer.layer_instance.bias"]=state_dict[f"flow.flows.{i}.enc.cond_layer.bias"]
        for j in range(3):
            params[f"flow.flows_{i}.enc.in_layers_{j}.layer_instance/kernel/scale"]=state_dict[f"flow.flows.{i}.enc.in_layers.{j}.weight_g"].squeeze()
            params[f"flow.flows_{i}.enc.in_layers_{j}.layer_instance.kernel"]=state_dict[f"flow.flows.{i}.enc.in_layers.{j}.weight_v"].transpose(0,2)
            params[f"flow.flows_{i}.enc.in_layers_{j}.layer_instance.bias"]=state_dict[f"flow.flows.{i}.enc.in_layers.{j}.bias"]
            params[f"flow.flows_{i}.enc.res_skip_layers_{j}.layer_instance/kernel/scale"]=state_dict[f"flow.flows.{i}.enc.res_skip_layers.{j}.weight_g"].squeeze()
            params[f"flow.flows_{i}.enc.res_skip_layers_{j}.layer_instance.kernel"]=state_dict[f"flow.flows.{i}.enc.res_skip_layers.{j}.weight_v"].transpose(0,2)
            params[f"flow.flows_{i}.enc.res_skip_layers_{j}.layer_instance.bias"]=state_dict[f"flow.flows.{i}.enc.res_skip_layers.{j}.bias"]



    params = {k: v.numpy() for k, v in params.items()}
    params = flax.traverse_util.unflatten_dict(params, sep=".")
    return params

